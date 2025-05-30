import plotly.colors as pc
import plotly.graph_objects as go
from itertools import cycle
import numpy as np
from copy import deepcopy
from .overlay import xyz2molecular_graph
from .data import atomic_number2element_symbol, atomic_number2hex
from .overlay import open_xyz_files, superimpose
from collections.abc import Iterable


def _plot_scale_box(box_half_length: float, unit_bar: bool = True):
    """
    Plot scale box and unit bar in 3D visualization.
    
    Parameters
    ----------
    box_half_length : float
        Half-length of the scale box.
    unit_bar : bool, optional
        Whether to show a 1Å unit bar or the full box length. Default is True.
    
    Returns
    -------
    list
        List of go.Scatter3d objects for the scale box and unit bar.
    """
    hl = box_half_length  # box half-length

    # Box vertices
    x_coords = [-hl, hl, hl, -hl, -hl, hl, hl, -hl]
    y_coords = [-hl, -hl, hl, hl, -hl, -hl, hl, hl]
    z_coords = [-hl, -hl, -hl, -hl, hl, hl, hl, hl]

    # Define edges by vertex indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    x_box_lines = []
    y_box_lines = []
    z_box_lines = []

    for edge in edges:
        for vertex in edge:
            x_box_lines.append(x_coords[vertex])
            y_box_lines.append(y_coords[vertex])
            z_box_lines.append(z_coords[vertex])
        x_box_lines.append(None)
        y_box_lines.append(None)
        z_box_lines.append(None)

    box_plots = [
        # Scaling bar
        go.Scatter3d(
            x=[hl, hl],
            y=[-hl, -hl+1 if unit_bar else +hl],
            z=[-hl, -hl],
            mode='lines+text',
            line=dict(color='red', width=13),
            text=['1 Å' if unit_bar else f'{np.round(hl*2, 1)} Å'],
            textposition='top center',
            textfont=dict(size=12, color='red'),
            showlegend=False,
            name='Bar',
            legendgroup='Box',
            hoverinfo='text',
            hovertext='Bar'
        ),
        # Box lines
        go.Scatter3d(
            x=x_box_lines,
            y=y_box_lines,
            z=z_box_lines,
            mode='lines',
            line=dict(color='grey', width=4),
            showlegend=True,
            name='Box',
            legendgroup='Box',
            hoverinfo='text',
            hovertext='Box'
        )
    ]
    return box_plots

def plot_overlay(xyz_format_jsons: list, colorby: str = "molecule", exclude_elements: list = None, 
                 exclude_atomic_idx: list = None, cmap: str = None, covalent_radius_percent: float = 108., **kwargs):
    """
    Visualizes molecular structures in 3D using Plotly.

    Parameters
    ----------
    xyz_format_jsons : list
        A list of dictionaries where each dictionary contains molecular data in JSON format with keys:
        - 'name': str, the name or identifier for the molecule.
        - 'n_atoms': int, the number of atoms in the molecule.
        - 'coordinate': ndarray, the atomic coordinates with columns for atomic number, x, y, z, and optionally index.
        - 'adjacency_matrix': ndarray, the matrix representing the connectivity of the molecule
        - 'bond_length_table': ndarray, the table of bond lengths with columns | atom_1_idx | atom_2_idx | distance |
    colorby : str, optional, default='molecule'
        Specifies how to color the molecules. Options:
        - 'molecule': Color by molecule.
        - 'atom': Color by element.
    exclude_elements : list, optional
        List of element symbols to exclude from visualization. e.g., ['H'] to exclude hydrogen.
    exclude_atomic_idx : list, optional
        List of atomic indices to exclude from visualization. Atomic index starts with 1. e.g. [1, 3, 4]
    cmap : str or list, optional
        str: A Plotly colormap name (e.g., 'Viridis', 'Plotly3') used to color the molecules.
        list: A list of color names or hex codes used for manual coloring (e.g., ['red', 'blue', 'green']).
    covalent_radius_percent : float, optional
        A percentage value used to scale the covalent radii for determining bonding (default is 108%).
    **kwargs
        Additional keyword arguments for customization:
        - alpha_atoms: float, optional, default=0.55 
            Opacity of atoms.
        - alpha_bonds: float, optional, default=0.35 
            Opacity of bonds.
        - atom_scaler: float, optional, default=4e1
            Scale factor for atom sphere radius.
        - bond_scaler: float, optional, default=7e4
            Scale factor for bond cylinder radius.
        - legend: bool, optional, default=False
            Whether to show legend.
        - show_index: bool, optional, default=False
            Whether to show atomic indices.
        - index_color: str, optional, default='red'
            Color of atomic indices.
        - index_size: int, optional, default=12
            Size of atomic indices text.
        - bgcolor: str, optional, default='black'
            Background color of the plot.
        - scale_box: bool, optional, default=True
            Whether to show the scale box.
        - unit_bar: bool, optional, default=True
            Whether to show a 1Å unit bar or the full box length.

    Returns
    -------
    None
        Displays the 3D plot using Plotly.
    """
    def _get_colors(cmap: str | list, n: int):
        """Get n size color list from Plotly colormap."""
        if not cmap:
            cmap = 'Plotly3'
        try:
            pc.get_colorscale(cmap)
        except Exception:
            print("\033[31m[WARNING]\033[0m", f"`{cmap}` is not a valid Plotly colormap. Applying default colormap instead.")
            cmap = 'Plotly3'
        if isinstance(cmap, str):
            colors = pc.get_colorscale(cmap)
            return list(pc.sample_colorscale(colors, [ratio for ratio in np.linspace(0, 1, n + 1)[1:]], colortype='rgb'))
        if isinstance(cmap, list):
            cyclic_iterator = cycle(cmap)
            return [next(cyclic_iterator) for _ in range(n)]

    # Set default values
    alpha_atoms = kwargs.get("alpha_atoms", 0.55)  # Atoms opacity
    alpha_bonds = kwargs.get("alpha_bonds", 0.35)  # Bonds opacity
    atom_scaler = kwargs.get("atom_scaler", 4e1)   # Sphere radius for atom view
    bond_scaler = kwargs.get("bond_scaler", 7e4)   # Cylinder radius for bond view
    legend = kwargs.get("legend", False)           # Show legend
    show_index = kwargs.get("show_index", False)   # Show atomic index
    index_color = kwargs.get("index_color", 'red') # Atomic index color
    index_size = kwargs.get("index_size", 12)      # Atomic index size
    bgcolor = kwargs.get("bgcolor", 'black')       # Background color
    scale_box = kwargs.get("scale_box", True)      # Show scale box
    unit_bar = kwargs.get("unit_bar", True)        # Show unit bar

    # Copy xyz_format_jsons
    _xyz_format_jsons = deepcopy(xyz_format_jsons)

    # Plotly figure
    fig = go.Figure()

    # Exclude atoms
    if exclude_atomic_idx:
        symbols_list = [xyz_json.get("coordinate")[:, 0] for xyz_json in _xyz_format_jsons]
        if not np.all(np.array(symbols_list) == symbols_list[0]):
            print("\033[31m[WARNING]\033[0m", "`exclude_atomic_idx` option expects that each coordinates has the same order of atoms")
        if 0 in exclude_atomic_idx:
            raise ValueError("atomic indices start with 1, but 0 was found in `exclude_atomic_idx`")
        exclude_atomic_idx = [idx - 1 for idx in exclude_atomic_idx]  # Reset atomic indices
        for mol_idx in range(len(_xyz_format_jsons)):
            atom_filtered_coordinate = [
                atomic_coordinate for atomic_idx, atomic_coordinate in enumerate(_xyz_format_jsons[mol_idx]["coordinate"])
                if atomic_idx not in exclude_atomic_idx
            ]
            _xyz_format_jsons[mol_idx]["coordinate"] = np.array(atom_filtered_coordinate)
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(atom_filtered_coordinate)

    # Exclude elements
    if exclude_elements:
        for mol_idx in range(len(_xyz_format_jsons)):
            element_filtered_coordinate = [
                atomic_coordinate for atomic_coordinate in _xyz_format_jsons[mol_idx]["coordinate"]
                if atomic_number2element_symbol[atomic_coordinate[0]] not in exclude_elements
            ]
            _xyz_format_jsons[mol_idx]["coordinate"] = np.array(element_filtered_coordinate)
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(element_filtered_coordinate)

    # Number of molecules
    num_of_xyz = len(_xyz_format_jsons)

    # Max number of atoms
    num_atom_xyz = max(len(xyz_jsons["coordinate"]) for xyz_jsons in _xyz_format_jsons)

    # Calculate global ranges for all molecular structures
    all_positions = np.vstack([xyz_json["coordinate"][:, 1:4] for xyz_json in _xyz_format_jsons])
    range_array = np.array([
        [np.min(all_positions[:, i]) for i in range(3)],
        [np.max(all_positions[:, i]) for i in range(3)]
    ])
    padding = 0.1
    if scale_box:
        half_box_length = np.max(np.abs(range_array)) * 1.3  # Scale box size based on max range
        max_range = [-half_box_length - padding, half_box_length + padding]
    else:
        max_range = [
            [range_array[0, i] - padding, range_array[1, i] + padding] for i in range(3)
        ]

    # Analyze molecular connectivity
    xyz2molecular_graph(_xyz_format_jsons, covalent_radius_percent)

    # Bond thickness and atom size
    bond_thickness = np.maximum(np.log10(bond_scaler / num_atom_xyz) * 2, 1)
    atom_size = np.maximum(np.log10(atom_scaler / num_atom_xyz) * 5, 2)

    if colorby == "molecule":
        palette = _get_colors(cmap, num_of_xyz)
        for mol_idx in range(len(_xyz_format_jsons)):
            color = palette[mol_idx]
            # Add atoms to plot
            fig.add_trace(go.Scatter3d(
                x=_xyz_format_jsons[mol_idx]["coordinate"][:, 1],
                y=_xyz_format_jsons[mol_idx]["coordinate"][:, 2],
                z=_xyz_format_jsons[mol_idx]["coordinate"][:, 3],
                mode='markers+text',
                opacity=alpha_atoms,
                marker=dict(size=atom_size, color=color),
                name=f'{_xyz_format_jsons[mol_idx]["name"]} atoms',
                text=_xyz_format_jsons[mol_idx]["coordinate"][:, 4].astype(int).astype(str) if show_index else None,
                textposition="top center" if show_index else None,
                textfont=dict(size=index_size, color=index_color) if show_index else None,
            ))
            legend_group_name = _xyz_format_jsons[mol_idx]["name"]
            # Add bonds to plot
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            first_bond = True
            for bond in bonds:
                bond = bond.astype(int) - 1
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[1]]
                fig.add_trace(go.Scatter3d(
                    x=[atom_1_coord[0], atom_2_coord[0]],
                    y=[atom_1_coord[1], atom_2_coord[1]],
                    z=[atom_1_coord[2], atom_2_coord[2]],
                    mode='lines',
                    opacity=alpha_bonds,
                    line=dict(width=bond_thickness, color=color),
                    legendgroup=legend_group_name,
                    name=f"{legend_group_name} bonds",
                    showlegend=True if first_bond else False
                ))
                first_bond = False

    elif colorby == "atom":
        if legend:
            print("\033[31m[WARNING]\033[0m", f"`legend`=True is not applicable when colorby='atom'.")
        if show_index:
            print("\033[31m[WARNING]\033[0m", f"`show_index`=True is not applicable when colorby='atom'.")
        all_coordinates = np.vstack([json["coordinate"] for json in _xyz_format_jsons])
        elements = set(all_coordinates[:, 0].astype(int))
        for element in elements:
            element_coordinates = all_coordinates[all_coordinates[:, 0].astype(int) == element]
            fig.add_trace(go.Scatter3d(
                x=element_coordinates[:, 1],
                y=element_coordinates[:, 2],
                z=element_coordinates[:, 3],
                mode='markers',
                opacity=alpha_atoms,
                marker=dict(size=atom_size, color=atomic_number2hex[element]),
                showlegend=False
            ))
        for mol_idx in range(len(_xyz_format_jsons)):
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            for bond in bonds:
                bond = bond.astype(int) - 1
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[1]]
                fig.add_trace(go.Scatter3d(
                    x=[atom_1_coord[0], atom_2_coord[0]],
                    y=[atom_1_coord[1], atom_2_coord[1]],
                    z=[atom_1_coord[2], atom_2_coord[2]],
                    mode='lines',
                    opacity=alpha_bonds,
                    line=dict(width=bond_thickness, color='gray'),
                    showlegend=False
                ))

    else:
        raise ValueError(f"Unsupported colorby: {colorby}")

    # Add scale box if enabled
    if scale_box:
        box_plots = _plot_scale_box(box_half_length=half_box_length, unit_bar=unit_bar)
        for plot in box_plots:
            fig.add_trace(plot)

    # Figure layout setting
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=max_range if scale_box else max_range[0], visible=False),
            yaxis=dict(range=max_range if scale_box else max_range[1], visible=False),
            zaxis=dict(range=max_range if scale_box else max_range[2], visible=False),
            aspectmode='manual' if scale_box else 'data',
            camera_projection=dict(type='orthographic'),
            aspectratio=dict(x=1, y=1, z=1) if scale_box else None
        ),
        showlegend=True if legend else False,
        paper_bgcolor=bgcolor
    )



    fig.show()


def plot_animation(xyz_format_jsons: list, colorby: str = "molecule", exclude_elements: list = None, 
                  exclude_atomic_idx: list = None, cmap: str = None, covalent_radius_percent: float = 108., **kwargs):
    """
    Visualization of molecular structures in 3D using Plotly with animation.

    Parameters
    ----------
    xyz_format_jsons : list
        List of dictionaries where each dictionary contains molecular data in JSON format with keys:
        - 'name': str, the name or identifier for the molecule.
        - 'n_atoms': int, the number of atoms in the molecule.
        - 'coordinate': ndarray, the atomic coordinates with columns for atomic number, x, y, z, and optionally index.
        - 'adjacency_matrix': ndarray, the matrix representing the connectivity of the molecule.
        - 'bond_length_table': ndarray, the table of bond lengths with columns | atom_1_idx | atom_2_idx | distance |
    colorby : str, optional, default='molecule'
        Specifies how to color the molecules. Options:
        - 'molecule': Color by molecule.
        - 'atom': Color by element.
    exclude_elements : list, optional
        List of element symbols to exclude from visualization. e.g., ['H']
    exclude_atomic_idx : list, optional
        List of atomic indices to exclude from visualization. Atomic index starts with 1. e.g. [1, 3, 4]
    cmap : str or list, optional
        str: A Plotly colormap name (e.g., 'Viridis', 'Plotly3') used to color the molecules.
        list: A list of color names or hex codes used for manual coloring (e.g., ['red', 'blue', 'green']).
    covalent_radius_percent : float, optional
        A percentage value used to scale the covalent radii for determining bonding (default is 108%).
    **kwargs
        Additional keyword arguments for customization:
        - alpha_atoms: float, optional, default=0.55
            Opacity of atoms.
        - alpha_bonds: float, optional, default=0.55
            Opacity of bonds.
        - atom_scaler: float, optional, default=2e1
            Scale factor for atom sphere radius.
        - bond_scaler: float, optional, default=1e4
            Scale factor for bond cylinder radius.
        - legend: bool, optional, default=False
            Whether to show legend.
        - scale_box: bool, optional, default=True
            Whether to show the scale box.
        - unit_bar: bool, optional, default=True
            Whether to show a 1Å unit bar or the full box length.

    Returns
    -------
    None
        Displays the 3D animated plot using Plotly.
    """
    STABLE = False
    if not STABLE:
        print("\033[31m[WARNING]\033[0m", f"plot_animation function is now testing. It is not working perfectly.")

    def _get_colors(cmap: str | list, n: int):
        """Get n size color list from Plotly colormap."""
        if not cmap:
            cmap = 'Plotly3'
        try:
            pc.get_colorscale(cmap)
        except Exception:
            print("\033[31m[WARNING]\033[0m", f"`{cmap}` is not a valid Plotly colormap. Applying default colormap instead.")
            cmap = 'Plotly3'
        if isinstance(cmap, str):
            colors = pc.get_colorscale(cmap)
            return list(pc.sample_colorscale(colors, [ratio for ratio in np.linspace(0, 1, n + 1)[1:]], colortype='rgb'))
        if isinstance(cmap, list):
            cyclic_iterator = cycle(cmap)
            return [next(cyclic_iterator) for _ in range(n)]

    # Set default values
    alpha_atoms = kwargs.get("alpha_atoms", 0.55)
    alpha_bonds = kwargs.get("alpha_bonds", 0.55)
    atom_scaler = kwargs.get("atom_scaler", 2e1)
    bond_scaler = kwargs.get("bond_scaler", 1e4)
    legend = kwargs.get("legend", False)
    scale_box = kwargs.get("scale_box", True)
    unit_bar = kwargs.get("unit_bar", True)

    # Validate input
    if not xyz_format_jsons:
        raise ValueError("xyz_format_jsons is empty. Please provide valid molecular data.")

    # Copy xyz_format_jsons
    _xyz_format_jsons = deepcopy(xyz_format_jsons)

    # Plotly figure
    fig = go.Figure()

    # Exclude atoms
    if exclude_atomic_idx:
        symbols_list = [xyz_json.get("coordinate")[:, 0] for xyz_json in _xyz_format_jsons]
        if not np.all(np.array(symbols_list) == symbols_list[0]):
            print("\033[31m[WARNING]\033[0m", "`exclude_atomic_idx` option expects that each coordinates has the same order of atoms")
        if 0 in exclude_atomic_idx:
            raise ValueError("atomic indices start with 1, but 0 was found in `exclude_atomic_idx`")
        exclude_atomic_idx = [idx - 1 for idx in exclude_atomic_idx]
        for mol_idx in range(len(_xyz_format_jsons)):
            atom_filtered_coordinate = [
                atomic_coordinate for atomic_idx, atomic_coordinate in enumerate(_xyz_format_jsons[mol_idx]["coordinate"])
                if atomic_idx not in exclude_atomic_idx
            ]
            _xyz_format_jsons[mol_idx]["coordinate"] = np.array(atom_filtered_coordinate)
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(atom_filtered_coordinate)

    # Exclude elements
    if exclude_elements:
        for mol_idx in range(len(_xyz_format_jsons)):
            element_filtered_coordinate = [
                atomic_coordinate for atomic_coordinate in _xyz_format_jsons[mol_idx]["coordinate"]
                if atomic_number2element_symbol[atomic_coordinate[0]] not in exclude_elements
            ]
            _xyz_format_jsons[mol_idx]["coordinate"] = np.array(element_filtered_coordinate)
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(element_filtered_coordinate)

    # Number of molecules
    num_of_xyz = len(_xyz_format_jsons)

    # Max number of atoms
    num_atom_xyz = max(len(xyz_jsons["coordinate"]) for xyz_jsons in _xyz_format_jsons)

    # Calculate global ranges for all molecular structures
    try:
        all_positions = np.vstack([xyz_json["coordinate"][:, 1:4] for xyz_json in _xyz_format_jsons])
        if all_positions.size == 0:
            raise ValueError("No valid coordinates found after filtering.")
        range_array = np.array([
            [np.min(all_positions[:, i]) for i in range(3)],
            [np.max(all_positions[:, i]) for i in range(3)]
        ])
    except Exception as e:
        raise ValueError(f"Failed to calculate coordinate ranges: {str(e)}")

    padding = 0.1
    if scale_box:
        half_box_length = np.max(np.abs(range_array)) * 1.3
        max_range = [-half_box_length - padding, half_box_length + padding]
    else:
        max_range = [
            [range_array[0, 0] - padding, range_array[1, 0] + padding],  # x
            [range_array[0, 1] - padding, range_array[1, 1] + padding],  # y
            [range_array[0, 2] - padding, range_array[1, 2] + padding]   # z
        ]

    # Analyze molecular connectivity
    xyz2molecular_graph(_xyz_format_jsons, covalent_radius_percent)

    # Bond thickness and atom size
    bond_thickness = np.maximum(np.log10(bond_scaler / num_atom_xyz) * 2, 1)
    atom_size = np.maximum(np.log10(atom_scaler / num_atom_xyz) * 5, 2)

    # Frame list to store animation frames
    frames = []

    if colorby == "molecule":
        palette = _get_colors(cmap, num_of_xyz)
        for mol_idx in range(len(_xyz_format_jsons)):
            frame_data = []
            color = palette[mol_idx]
            # Add atoms
            frame_data.append(go.Scatter3d(
                x=_xyz_format_jsons[mol_idx]["coordinate"][:, 1],
                y=_xyz_format_jsons[mol_idx]["coordinate"][:, 2],
                z=_xyz_format_jsons[mol_idx]["coordinate"][:, 3],
                mode='markers',
                opacity=alpha_atoms,
                marker=dict(size=atom_size, color=color),
                name=f'{_xyz_format_jsons[mol_idx]["name"]} atoms'
            ))
            # Add bonds
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            bond_x, bond_y, bond_z = [], [], []
            for bond in bonds:
                bond = bond.astype(int) - 1
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[1]]
                bond_x.extend([atom_1_coord[0], atom_2_coord[0], None])
                bond_y.extend([atom_1_coord[1], atom_2_coord[1], None])
                bond_z.extend([atom_1_coord[2], atom_2_coord[2], None])
            frame_data.append(go.Scatter3d(
                x=bond_x, y=bond_y, z=bond_z,
                mode='lines',
                opacity=alpha_bonds,
                line=dict(width=bond_thickness, color=color),
                name=f"{_xyz_format_jsons[mol_idx]['name']} bonds"
            ))
            # Add scale box if enabled
            if scale_box:
                frame_data.extend(_plot_scale_box(box_half_length=half_box_length, unit_bar=unit_bar))
            frames.append(go.Frame(data=frame_data, name=str(mol_idx)))

    elif colorby == "atom":
        if legend:
            print("\033[31m[WARNING]\033[0m", f"`legend`=True is not applicable when colorby='atom'.")
        for mol_idx in range(len(_xyz_format_jsons)):
            frame_data = []
            # Add atoms by element
            elements = set(_xyz_format_jsons[mol_idx]["coordinate"][:, 0].astype(int))
            for element in elements:
                element_coordinates = _xyz_format_jsons[mol_idx]["coordinate"][
                    _xyz_format_jsons[mol_idx]["coordinate"][:, 0].astype(int) == element
                ]
                frame_data.append(go.Scatter3d(
                    x=element_coordinates[:, 1],
                    y=element_coordinates[:, 2],
                    z=element_coordinates[:, 3],
                    mode='markers',
                    opacity=alpha_atoms,
                    marker=dict(size=atom_size, color=atomic_number2hex[element]),
                    name=f'Element {atomic_number2element_symbol[element]}',
                    showlegend=False
                ))
            # Add bonds
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            bond_x, bond_y, bond_z = [], [], []
            for bond in bonds:
                bond = bond.astype(int) - 1
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[1]]
                bond_x.extend([atom_1_coord[0], atom_2_coord[0], None])
                bond_y.extend([atom_1_coord[1], atom_2_coord[1], None])
                bond_z.extend([atom_1_coord[2], atom_2_coord[2], None])
            frame_data.append(go.Scatter3d(
                x=bond_x, y=bond_y, z=bond_z,
                mode='lines',
                opacity=alpha_bonds,
                line=dict(width=bond_thickness, color='gray'),
                name=f"{_xyz_format_jsons[mol_idx]['name']} bonds",
                showlegend=False
            ))
            # Add scale box if enabled
            if scale_box:
                frame_data.extend(_plot_scale_box(box_half_length=half_box_length, unit_bar=unit_bar))
            frames.append(go.Frame(data=frame_data, name=str(mol_idx)))

    else:
        raise ValueError(f"Unsupported colorby: {colorby}")

    # Set initial frame
    fig.add_traces(frames[0].data)

    # Layout settings
    fig.frames = frames
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 30, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate',
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate',
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}], 'label': str(k), 'method': 'animate'} for k, f in enumerate(fig.frames)]
        }],
        scene=dict(
            xaxis=dict(range=max_range if scale_box else max_range[0], visible=False),
            yaxis=dict(range=max_range if scale_box else max_range[1], visible=False),
            zaxis=dict(range=max_range if scale_box else max_range[2], visible=False),
            aspectmode='manual' if scale_box else 'data',
            camera_projection=dict(type='orthographic'),
            aspectratio=dict(x=1, y=1, z=1) if scale_box else None
        ),
        showlegend=True if legend else False
    )

    fig.show()


class Parameters:
    """
    A simple class to manage parameters, allowing dictionary keys to be accessed like attributes.

    Example
    -------
    >>> params = Parameters({
            "cmap": 'Plotly3',
            "legend": False,
            "bgcolor": 'black'
        })
    >>> print(params.cmap)
    'Plotly3'
    >>> print(params.legend)
    False
    >>> params.legend = True
    >>> print(params)
    {'cmap': 'Plotly3', 'legend': True, 'bgcolor': 'black'}
    """
    def __init__(self, parameters):
        self._parameters = parameters

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        raise AttributeError(f"parameters has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_parameters":
            super().__setattr__(name, value)
        else:
            self._parameters[name] = value

    def __repr__(self):
        from pprint import pformat
        return pformat(self._parameters, indent=0)

class OverlayMolecules:
    """
    A class to overlay molecules with customizable parameters.

    Attributes
    ----------
    parameters : Parameters
        A Parameters instance containing the configuration for overlaying molecules.

    xyz_format_jsons : list
        A list of JSON formatted molecular data.

    superimposed_jsons : list
        A list of superimposed molecular data based on the configured options.

    Methods
    -------
    set_molecules(filenames)
        Opens and reads XYZ files, extracting headers and atomic coordinates.

    add_molecules(filenames)
    	Add molecules.

    plot_overlay()
        Visualizes the superimposed molecular structures in 3D using Plotly.

    plot_animation()
        (Currently in development) Creates an animated visualization of molecular structures.
    
    Example
    -------
    >>> # config molecules
    >>> overlayplot = OverlayMolecules(filenames=['molecule1.xyz', 'molecule2.xyz'])
    >>> 
    >>> # adjust options
    >>> overlayplot.parameters.cmap = 'Viridis'
    >>> overlayplot.parameters.legend = True
    >>>
    >>> # plot
    >>> overlayplot.plot_overlay()
    >>> overlayplot.plot_animation()  # Note: This is under development
    """
    np.set_printoptions(precision=7, suppress=True)

    # default parameters
    default_parameters = {
        # coordinates
        "superimpose_option": "aa",
        "superimpose_option_param": None,

        # visualization setting
        "colorby": "molecule",
        "exclude_elements": None,
        "exclude_atomic_idx": None,
        "covalent_radius_percent": 108.,
        "scaled_box": True,

        # plot setting
        "alpha_atoms": 0.55,
        "alpha_bonds": 0.35,
        "atom_scaler": 4e1,
        "bond_scaler": 7e4,
        "cmap": 'Plotly3',
        "legend": False,
        "bgcolor": 'black',
        "show_index": False,
        "index_color": "red",
        "index_size": 12
    }

    def __init__(self, filenames:Iterable|str=None, **kwargs):
        """
        Initializes the OverlayMolecules instance with optional file names and parameters.

        Parameters
        ----------
        filenames : str or list of str, optional
            File paths or list of file paths to XYZ files to initialize with.
        **kwargs : keyword arguments
            Optional parameters to customize the overlay behavior.
        """
        # default parameters
        self.parameters = Parameters({**OverlayMolecules.default_parameters, **kwargs})

        self.xyz_format_jsons = None
        self.superimposed_jsons = None
        
        # config molecules
        if filenames is not None:
            self.set_molecules(filenames=filenames)
        
    def set_molecules(self, filenames:Iterable|str):
        '''
        Opens and reads XYZ files, extracting atomic coordinates.

        Parameters
        ----------
        filenames : str or list
            str : xyz format traj file path
            list : list of xyz format strings or file paths

        Usage
        -----
        >>> myclass = OverlayMolecules()
        >>>
        >>> # list of dictinoaries #
        >>> xyz_files = [
        >>>     {'name': 'reactant', 'coordinate': 'sn2_reac.xyz'},
        >>>     {'name': 'TS', 'coordinate': 'sn2_TS.xyz'},
        >>>     {'name': 'prod', 'coordinate': 'sn2_prod'}
        >>>  ]
        >>> myclass.set_molecules(xyz_files)
        >>> 
        >>> # list of strings #
        >>> myclass.set_molecules(['reac.xyz', 'TS.xyz', 'prod.xyz'])
        >>> 
        >>> # traj file #
        >>> myclass.set_molecules('sn2_traj.xyz')
        '''
        self.xyz_format_jsons = open_xyz_files(filenames)
        self.superimposed_jsons = self._superimpose(self.xyz_format_jsons)

    def add_molecules(self, filenames:Iterable|str):
        '''
        Opens and reads XYZ files, extracting atomic coordinates.

        Parameters
        ----------
        filenames : str or list
            str : xyz format traj file path
            list : list of xyz format strings or file paths
	    '''
        new_xyz_format_jsons = open_xyz_files(filenames)
        self.xyz_format_jsons.extend(new_xyz_format_jsons)
        self.superimposed_jsons = self._superimpose(self.xyz_format_jsons)
	
	
    def _superimpose(self, xyz_format_jsons):
        """
        Superimposes molecular coordinates based on the configured options.

        Parameters
        ----------
        xyz_format_jsons : list of dict
            List of JSON formatted molecular data.

        Returns
        -------
        list of dict
            Superimposed molecular data.
        """
        return superimpose(xyz_format_jsons, self.parameters.superimpose_option, self.parameters.superimpose_option_param)

    
    def plot_overlay(self):
        """Visualizes the superimposed molecular structures in 3D using Plotly.
        """
        if self.superimposed_jsons == None: raise ValueError("Please set molecules first.")

        plot_overlay(
            xyz_format_jsons=self.superimposed_jsons,
            colorby=self.parameters.colorby,
            exclude_elements=self.parameters.exclude_elements,
            exclude_atomic_idx=self.parameters.exclude_atomic_idx,
            cmap=self.parameters.cmap,
            covalent_radius_percent=self.parameters.covalent_radius_percent,
            alpha_atoms=self.parameters.alpha_atoms,
            alpha_bonds=self.parameters.alpha_bonds,
            atom_scaler=self.parameters.atom_scaler,
            bond_scaler=self.parameters.bond_scaler,
            legend=self.parameters.legend,
            bgcolor=self.parameters.bgcolor,
            show_index=self.parameters.show_index,
            index_color=self.parameters.index_color,
            index_size=self.parameters.index_size,
            scaled_box=self.parameters.scaled_box

            )
        
    def plot_animation(self):
        """
        Now in development
        """
        if self.superimposed_jsons == None: raise ValueError("Please set molecules first.")

        plot_animation(
            xyz_format_jsons=self.superimposed_jsons,
            colorby=self.parameters.colorby,
            exclude_elements=self.parameters.exclude_elements,
            exclude_atomic_idx=self.parameters.exclude_atomic_idx,
            cmap=self.parameters.cmap,
            covalent_radius_percent=self.parameters.covalent_radius_percent,
            alpha_atoms=self.parameters.alpha_atoms,
            alpha_bonds=self.parameters.alpha_bonds,
            atom_scaler=self.parameters.atom_scaler,
            bond_scaler=self.parameters.bond_scaler,
            legend=self.parameters.legend,
            scaled_box=self.parameters.scaled_box,
        )
