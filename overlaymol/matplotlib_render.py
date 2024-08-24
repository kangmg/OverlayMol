import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from copy import deepcopy 
from mpl_toolkits.mplot3d import Axes3D
from .overlay import xyz2molecular_graph
from .data import atomic_number2element_symbol, atomic_number2hex

def set_axes_equal(ax)->Axes3D:
    """
    Adjust the scaling of a 3D plot so that all axes are equally proportioned, 
    ensuring that geometric shapes (e.g., spheres, cubes) maintain their correct proportions.

    Parameters
    ----------
    ax : Axes3D
        The Matplotlib 3D axis object to be adjusted.

    Note
    ----
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """
    # Get current limits of the axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate ranges and midpoints for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # Calculate the plot radius (half the max range, scaled)
    plot_radius = 0.35 * max([x_range, y_range, z_range])

    # Set new limits for each axis centered around the midpoints
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_overlay(xyz_format_jsons:list, colorby:str="molecule", exclude_elements:list=None, exclude_atomic_idx:list=None, cmap:str|list=None, covalent_radius_percent:float=108., **kwargs):
    """
    Visualizes molecular structures in 3D using Plotly.

    Parameters
    ----------
    xyz_format_jsons : list
        A list of dictionaries where each dictionary contains molecular data in JSON format with keys:
        - "name": str, the name or identifier for the molecule.
        - "n_atoms": int, the number of atoms in the molecule.
        - "coordinate": ndarray, the atomic coordinates with columns for atomic number, x, y, z, and optionally index.
        - "adjacency_matrix": ndarray, the matrix representing the connectivity of the molecule
        - "bond_length_table": ndarray, the table of bond lengths with columns | atom_1_idx | atom_2_idx | distance |

    colorby : str, optional, default="molecule"
        Specifies how to color the molecules. Options:
        - "molecule": Color by molecule.
        - "atom": Color by element.
	
    exclude_elements : list, optional
        List of element symbols to exclude from visualization. e.g., ["H"] to exclude hydrogen.
	
    exclude_atomic_idx : list, optional
        List of atomic indices to exclude from visualization. Atomic index starts with 1. e.g. [1, 3, 4]
	
	cmap : str or list, optional
		str : A Matplotlib colormap name (e.g., 'Greys', 'Purples', 'Blues' etc..) used to color the molecules.
			- Refer to the Matplotlib documentation : https://matplotlib.org/stable/users/explain/colors/colormaps.html
		list : A list of color names for manual coloring (e.g., ['b', 'g', 'r']).
	
	covalent_radius_percent : float, optional
		A percentage value used to scale the covalent radii for determining bonding (default is 108%).
	
    **kwargs
        Additional keyword arguments for customization:
        - alpha_atoms: float, optional, default=0.55, opacity of atoms.
        - alpha_bonds: float, optional, default=0.35, opacity of bonds.
        - atom_scaler: float, optional, default=4e1, scale factor for atom sphere radius.
        - bond_scaler: float, optional, default=7e4, scale factor for bond cylinder radius.
        - legend: bool, optional, default=False, whether to show legend.
    """
    def _get_cmap(cmap:str|list):
        """set matplotlib colormap
        """
        # use matplotlib colormap
        if not cmap:
            cmap = "Blues"
            #cmap = ['b', 'r', 'g']

        if isinstance(cmap, str):
            try: plt.get_cmap(cmap)
            except ValueError:
                print("\033[31m[WARNING]\033[0m", f"`{cmap}` is not a valid matplotlib colormap. Applying default colormap instead.")

            # get color palette
            palette = list(plt.get_cmap(cmap)(ratio) for ratio in np.linspace(0, 1, num_of_xyz+1)[1:])
            color_cycle = cycle(palette)

        if isinstance(cmap, list):
            color_cycle = cycle(cmap)

        return color_cycle

    # set default values
    alpha_atoms = kwargs.get("alpha_atom", 0.55) # atoms opacity
    alpha_bonds = kwargs.get("alpha_bond", 0.55) # bonds opacity
    atom_scaler = kwargs.get("atom_scaler", 1e60) # sphere radius for atom view, change exponent
    bond_scaler = kwargs.get("bond_scaler", 1e6) # cylinder radius for bond view, change exponent
    legend = kwargs.get("legend", False) # add legend

    # copy xyz_format_jsons
    _xyz_format_jsons = deepcopy(xyz_format_jsons)

    # plt figure setting
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()

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

    if colorby=="molecule":
        # number of molecules
        num_of_xyz = len(_xyz_format_jsons)
        # max number of atoms
        num_atom_xyz = max(len(xyz_jsons["coordinate"]) for xyz_jsons in _xyz_format_jsons)

        # set color map
        color_cycle = _get_cmap(cmap)

        # analyze molecular connectivity
        xyz2molecular_graph(_xyz_format_jsons, covalent_radius_percent)

        # plot atoms & bonds
        for mol_idx in range(len(_xyz_format_jsons)):
            color = next(color_cycle)
            # plot atoms
            ax.scatter(*_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4].T,
                       s=np.log10(atom_scaler/num_atom_xyz),
                       alpha=alpha_atoms,
                       c=color,
                       label=_xyz_format_jsons[mol_idx]["name"])


            # plot bonds
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            for bond in bonds:
                bond = bond.astype(int) - 1 # internally idx start with 0
                # convert symbol to atomic nubmer
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[1]]
                ax.plot(*np.array([atom_1_coord, atom_2_coord]).T,
                        linewidth=np.log10(bond_scaler/num_atom_xyz),
                        alpha=alpha_bonds,
                        c=color)

    elif colorby=="atom":

        if legend: print("\033[31m[WARNING]\033[0m", f"`legend`=True is not a applicable when colorby='atom'.")

        # number of molecules
        num_of_xyz = len(_xyz_format_jsons)
        # max number of atoms
        num_atom_xyz = max(len(xyz_jsons["coordinate"]) for xyz_jsons in _xyz_format_jsons)

        # analyze molecular connectivity
        xyz2molecular_graph(_xyz_format_jsons, covalent_radius_percent)

        # plot atoms
        all_coordinates = np.vstack(list(json["coordinate"] for json in _xyz_format_jsons))
        elements = set(all_coordinates[:, 0].astype(int))
        # plot element-wise
        for element in elements:
            element_coordinates = all_coordinates[all_coordinates[:, 0].astype(int) == element]
            ax.scatter(*element_coordinates[:, 1:4].T, s=np.log10(atom_scaler/num_atom_xyz), alpha=alpha_atoms, c=atomic_number2hex[element])

        # plot bonds
        for mol_idx in range(len(_xyz_format_jsons)):
            bonds = _xyz_format_jsons[mol_idx]["bond_length_table"][:, :2]
            for bond in bonds:
                bond = bond.astype(int) - 1 # internally idx start with 0
                # convert symbol to atomic nubmer
                atom_1_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[0]]
                atom_2_coord = _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][bond[1]]
                ax.plot(*np.array([atom_1_coord, atom_2_coord]).T, linewidth=np.log10(bond_scaler/num_atom_xyz), alpha=alpha_bonds, c="grey")

    else:
        raise ValueError(f"Unsupported option : {colorby}")

    set_axes_equal(ax) # adjust 3d drawing behavior, otherwise molecules are not correctly displayes
    #show the plot
    if legend: ax.legend()
    plt.show()
