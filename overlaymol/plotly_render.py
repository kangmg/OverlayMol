import plotly.colors as pc
import plotly.graph_objects as go
from itertools import cycle
import numpy as np
from copy import deepcopy
from .overlay import xyz2molecular_graph
from .data import atomic_number2element_symbol, atomic_number2hex


def plot_overlay(xyz_format_jsons:list, colorby:str="molecule", exclude_elements:list=None, exclude_atomic_idx:list=None, cmap:str=None, covalent_radius_percent:float=108., **kwargs):
    """
    Description
    -----------
    Visualization of molecular structures in 3D using Plotly.

    Parameters
    ----------
    - xyz_format_jsons : list
        list of json format xyz

    - colorby : str
        supported options : ["molecule", "atom"]
        - molecule  : color by molecule
        - atom      : color by atom

    - exclude_elements : list
        list of elements to exclude from visualization. e.g. ["H"]

    - exclude_atomic_idx : list
        list of atoms to exclude from visualization. e.g. [1, 3, 4]

    - cmap : str or list
        plotly colormap to use for coloring.
        Supported options : [  ]
        Refer)
        https://plotly.com/python/builtin-colorscales/

        or

        iterable color list
        e.g. ['red', 'blue', 'green']
        Refer)
        https://community.plotly.com/t/plotly-colours-list/11730/3


    - covalent_radius_percent : float
        resize covalent radii by this percent
        default : 108%

    Returns
    -------
    """
    def _get_colors(cmap:str|list, n:int):
        """get n size color list from plotly colormap
        """
        if not cmap:
            cmap = 'Plotly3'

        try: pc.get_colorscale(cmap)
        except Exception:
            print("\033[31m[WARNING]\033[0m", f"`{cmap}` is not a valid plotly colormap. Applying default colormap instead.")
            cmap = 'Plotly3'

        if isinstance(cmap, str):
            colors = pc.get_colorscale(cmap)
            return list(pc.sample_colorscale(colors, list(ratio for ratio in np.linspace(0, 1, n+1)[1:]), colortype='rgb'))

        if isinstance(cmap, list):
            cyclic_iterator = cycle(cmap)
            return list(next(cyclic_iterator) for _ in range(n))

    # set default values
    alpha_atoms = kwargs.get("alpha_atoms", 0.55) # atoms opacity
    alpha_bonds = kwargs.get("alpha_bonds", 0.35) # bonds opacity
    atom_scaler = kwargs.get("atom_scaler", 4e1) # sphere radius for atom view, change exponent
    bond_scaler = kwargs.get("bond_scaler", 7e4) # cylinder radius for bond view, change exponent
    legend = kwargs.get("legend", False) # add legend
    show_index = kwargs.get("show_index", False) # show atomic index
    index_color = kwargs.get("index_color", 'red') # atomic index color
    index_size = kwargs.get("index_size", 12) # atomic index size
    bgcolor = kwargs.get("bgcolor", 'black') # background color

    # copy xyz_format_jsons
    _xyz_format_jsons = deepcopy(xyz_format_jsons)

    # plotly figure
    fig = go.Figure()

    # exclude atoms
    if exclude_atomic_idx:
        # `exclude_atomic_idx` option expects that each coordinates has the same order of atoms
        symbols_list = list(map(lambda xyz_json : xyz_json.get("coordinate")[:, 0], _xyz_format_jsons))
        if not np.all(np.array(symbols_list) == symbols_list[0]):
            print("\033[31m[WARNING]\033[0m", "`exclude_atomic_idx` option expects that each coordinates has the same order of atoms")
        # atomic indice start with 1
        if 0 in exclude_atomic_idx: raise ValueError("atomic indices start with 1, but 0 was found in `exclude_atomic_idx`")

        # reset atomic indice
        exclude_atomic_idx = list(idx - 1 for idx in exclude_atomic_idx)

        # check if atomic index is out of range
        if any(max(exclude_atomic_idx) > len(_xyz_format_jsons[mol_idx]["coordinate"]) for mol_idx in range(len(_xyz_format_jsons))):
            raise ValueError(f"Atomic index {max(exclude_atomic_idx)} provided in `exclude_atomic_idx` is out of range in your molecule.")

        for mol_idx in range(len(_xyz_format_jsons)):
            # filter the atom in `exclude_atomic_idx`
            atom_filtered_coordinate = list(
                atomic_coordinate for atomic_idx, atomic_coordinate in enumerate(_xyz_format_jsons[mol_idx]["coordinate"]) if atomic_idx not in exclude_atomic_idx
                  )
            # overwrite filtered coordinate
            _xyz_format_jsons[mol_idx]["coordinate"] = atom_filtered_coordinate
            # adjust number of atoms : n_atoms
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(atom_filtered_coordinate)

    # exclude elements
    if exclude_elements:
        for mol_idx in range(len(_xyz_format_jsons)):
            # filter the element in `exclude_elements`
            element_filtered_coordinate = list(
                atomic_coordinate for atomic_coordinate in _xyz_format_jsons[mol_idx]["coordinate"] if atomic_number2element_symbol[atomic_coordinate[0]] not in exclude_elements
                  )
            # overwrite filtered coordinate
            _xyz_format_jsons[mol_idx]["coordinate"] = element_filtered_coordinate
            # adjust number of atoms : n_atoms
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(element_filtered_coordinate)


    # number of molecules
    num_of_xyz = len(_xyz_format_jsons)

    # max number of atoms
    num_atom_xyz = max(len(xyz_jsons["coordinate"]) for xyz_jsons in _xyz_format_jsons)

    # analyze molecular connectivity
    xyz2molecular_graph(_xyz_format_jsons, covalent_radius_percent)

    # bond thickness and atom size
    bond_thickness = np.maximum(np.log10(bond_scaler / num_atom_xyz) * 2, 1)
    atom_size = np.maximum(np.log10(atom_scaler / num_atom_xyz) * 5, 2)

    if colorby == "molecule":
        # set color palette
        palette = _get_colors(cmap, num_of_xyz)

        # plot atoms & bonds
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
                #name=_xyz_format_jsons[mol_idx]["name"]
                name=f'{_xyz_format_jsons[mol_idx]["name"]} atoms',
                text=_xyz_format_jsons[mol_idx]["coordinate"][:, 4].astype(int).astype(str) if show_index else None,
                textposition="top center" if show_index else None,
                textfont=dict(
                    size=index_size,
                    color=index_color
                    ) if show_index else None,
                ))
            legend_group_namegroup = _xyz_format_jsons[mol_idx]["name"]

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
                    legendgroup=legend_group_namegroup,
                    name=f"{legend_group_namegroup} bonds",
                    showlegend=True if first_bond else False
                ))
                first_bond = False


    elif colorby == "atom":
        # `legend` is only working when colorby='molecule'.
        if legend: print("\033[31m[WARNING]\033[0m", f"`legend`=True is not a applicable when colorby='atom'.")
        if show_index: print("\033[31m[WARNING]\033[0m", f"`show_indice`=True is not a applicable when colorby='atom'.")

        # plot atoms
        all_coordinates = np.vstack(list(json["coordinate"] for json in _xyz_format_jsons))
        elements = set(all_coordinates[:, 0].astype(int))

        # plot element-wise
        for element in elements:
            element_coordinates = all_coordinates[all_coordinates[:, 0].astype(int) == element]

            # Add atoms to plot
            fig.add_trace(go.Scatter3d(
                x=element_coordinates[:, 1],
                y=element_coordinates[:, 2],
                z=element_coordinates[:, 3],
                mode='markers',
                opacity=alpha_atoms,
                marker=dict(size=atom_size, color=atomic_number2hex[element]),
                showlegend=False
            ))

        # plot bonds
        for mol_idx in range(len(_xyz_format_jsons)):
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            for bond in bonds:
                bond = bond.astype(int) - 1 # internally idx start with 0
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:][bond[1]]
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
        raise ValueError(f"Unsupported colorby : {colorby}")

    # figure layout setting
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera_projection=dict(type='orthographic'),
            #aspectratio=dict(x=1, y=1, z=1)
        ),
        showlegend=True if legend else False,
        paper_bgcolor=bgcolor
    )

    fig.show()




def plot_animation(xyz_format_jsons:list, colorby:str="molecule", exclude_elements:list=None, exclude_atomic_idx:list=None, cmap:str=None, covalent_radius_percent:float=108., **kwargs):
    """
    Description
    -----------
    Visualization of molecular structures in 3D using Plotly.

    Parameters
    ----------
    - xyz_format_jsons : list
        list of json format xyz

    - colorby : str
        supported options : ["molecule", "atom"]
        - molecule  : color by molecule
        - atom      : color by atom

    - exclude_elements : list
        list of elements to exclude from visualization. e.g. ["H"]

    - exclude_atomic_idx : list
        list of atoms to exclude from visualization. e.g. [1, 3, 4]

    - cmap : str or list
        plotly colormap to use for coloring.
        Supported options : [  ]
        Refer)
        https://plotly.com/python/builtin-colorscales/

        or

        iterable color list
        e.g. ['red', 'blue', 'green']
        Refer)
        https://community.plotly.com/t/plotly-colours-list/11730/3


    - covalent_radius_percent : float
        resize covalent radii by this percent
        default : 108%

    Returns
    -------
    """
    STABLE = False
    if not STABLE:
        print("\033[31m[WARNING]\033[0m", f"plot_animation function is now testing. It is not working perfectly.")

    def _get_colors(cmap:str|list, n:int):
        """get n size color list from plotly colormap
        """
        if not cmap:
            cmap = 'Plotly3'

        try: pc.get_colorscale(cmap)
        except Exception:
            print("\033[31m[WARNING]\033[0m", f"`{cmap}` is not a valid plotly colormap. Applying default colormap instead.")
            cmap = 'Plotly3'

        if isinstance(cmap, str):
            colors = pc.get_colorscale(cmap)
            return list(pc.sample_colorscale(colors, list(ratio for ratio in np.linspace(0, 1, n+1)[1:]), colortype='rgb'))

        if isinstance(cmap, list):
            cyclic_iterator = cycle(cmap)
            return list(next(cyclic_iterator) for _ in range(n))

    # set default values
    alpha_atoms = kwargs.get("alpha_atoms", 0.55) # atoms opacity
    alpha_bonds = kwargs.get("alpha_bonds", 0.55) # bonds opacity
    atom_scaler = kwargs.get("atom_scaler", 2e1) # sphere radius for atom view, change exponent
    bond_scaler = kwargs.get("bond_scaler", 1e4) # cylinder radius for bond view, change exponent
    legend = kwargs.get("legend", False) # add legend

    # copy xyz_format_jsons
    _xyz_format_jsons = deepcopy(xyz_format_jsons)

    # plotly figure
    fig = go.Figure()

    # exclude atoms
    if exclude_atomic_idx:
        # `exclude_atomic_idx` option expects that each coordinates has the same order of atoms
        symbols_list = list(map(lambda xyz_json : xyz_json.get("coordinate")[:, 0], _xyz_format_jsons))
        if not np.all(np.array(symbols_list) == symbols_list[0]):
            print("\033[31m[WARNING]\033[0m", "`exclude_atomic_idx` option expects that each coordinates has the same order of atoms")
        # atomic indice start with 1
        if 0 in exclude_atomic_idx: raise ValueError("atomic indices start with 1, but 0 was found in `exclude_atomic_idx`")

        # reset atomic indice
        exclude_atomic_idx = list(idx - 1 for idx in exclude_atomic_idx)

        # check if atomic index is out of range
        if any(max(exclude_atomic_idx) > len(_xyz_format_jsons[mol_idx]["coordinate"]) for mol_idx in range(len(_xyz_format_jsons))):
            raise ValueError(f"Atomic index {max(exclude_atomic_idx)} provided in `exclude_atomic_idx` is out of range in your molecule.")

        for mol_idx in range(len(_xyz_format_jsons)):
            # filter the atom in `exclude_atomic_idx`
            atom_filtered_coordinate = list(
                atomic_coordinate for atomic_idx, atomic_coordinate in enumerate(_xyz_format_jsons[mol_idx]["coordinate"]) if atomic_idx not in exclude_atomic_idx
                  )
            # overwrite filtered coordinate
            _xyz_format_jsons[mol_idx]["coordinate"] = atom_filtered_coordinate
            # adjust number of atoms : n_atoms
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(atom_filtered_coordinate)

    # exclude elements
    if exclude_elements:
        for mol_idx in range(len(_xyz_format_jsons)):
            # filter the element in `exclude_elements`
            element_filtered_coordinate = list(
                atomic_coordinate for atomic_coordinate in _xyz_format_jsons[mol_idx]["coordinate"] if atomic_number2element_symbol[atomic_coordinate[0]] not in exclude_elements
                  )
            # overwrite filtered coordinate
            _xyz_format_jsons[mol_idx]["coordinate"] = element_filtered_coordinate
            # adjust number of atoms : n_atoms
            _xyz_format_jsons[mol_idx]["n_atoms"] = len(element_filtered_coordinate)


    # 모든 분자 구조에 대한 전체 범위 계산
    all_coords = np.vstack([xyz["coordinate"][:, 1:] for xyz in _xyz_format_jsons])
    x_range = [np.min(all_coords[:, 0]), np.max(all_coords[:, 0])]
    y_range = [np.min(all_coords[:, 1]), np.max(all_coords[:, 1])]
    z_range = [np.min(all_coords[:, 2]), np.max(all_coords[:, 2])]

    # 약간의 여유 추가
    padding = 0.1
    x_range = [x_range[0] - padding, x_range[1] + padding]
    y_range = [y_range[0] - padding, y_range[1] + padding]
    z_range = [z_range[0] - padding, z_range[1] + padding]

    # 프레임 리스트를 저장할 변수
    frames = []

    if colorby == "molecule":
        # number of molecules
        num_of_xyz = len(_xyz_format_jsons)
        # max number of atoms
        num_atom_xyz = max(len(xyz_jsons["coordinate"]) for xyz_jsons in _xyz_format_jsons)

        # set color palette
        palette = _get_colors(cmap, num_of_xyz)

        # analyze molecular connectivity
        xyz2molecular_graph(_xyz_format_jsons, covalent_radius_percent)


        # 각 분자에 대해 프레임 생성
        for mol_idx in range(len(_xyz_format_jsons)):
            frame_data = []
            color = palette[mol_idx]

            # 원자 추가
            atom_size = np.maximum(np.log10(atom_scaler / num_atom_xyz) * 5, 2)
            frame_data.append(go.Scatter3d(
                x=_xyz_format_jsons[mol_idx]["coordinate"][:, 1],
                y=_xyz_format_jsons[mol_idx]["coordinate"][:, 2],
                z=_xyz_format_jsons[mol_idx]["coordinate"][:, 3],
                mode='markers',
                opacity=alpha_atoms,
                marker=dict(size=atom_size, color=color),
                name=f'{_xyz_format_jsons[mol_idx]["name"]} atoms'
            ))

            # 결합 추가
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            bond_thickness = np.maximum(np.log10(bond_scaler / num_atom_xyz) * 2, 1)
            bond_x, bond_y, bond_z = [], [], []
            for bond in bonds:
                bond = bond.astype(int) - 1
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:][bond[1]]
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

            # 프레임 추가
            frames.append(go.Frame(data=frame_data, name=str(mol_idx)))

        # 초기 프레임 설정
        fig.add_traces(frames[0].data)

    # 레이아웃 설정
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
            xaxis=dict(range=x_range , visible=False),
            yaxis=dict(range=y_range , visible=False),
            zaxis=dict(range=z_range , visible=False),
            aspectmode='data',
            camera_projection=dict(type='orthographic'),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        showlegend=True if legend else False
    )

    fig.show()