from numpy import ndarray
import numpy as np
from os.path import isfile, basename
import re
from collections.abc import Iterable
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from .data import element_symbol2atomic_number, covalent_radii, atomic_number2element_symbol


def kabsch(P, Q):
    """
    Compute the optimal rotation matrix using the Kabsch algorithm to align
    two sets of points P and Q.
    
    Ref. https://github.com/charnley/rmsd

    Parameters
    ----------
    P : ndarray
        An array of shape (N, 3) representing the first set of points.
    Q : ndarray
        An array of shape (N, 3) representing the second set of points.

    Returns
    -------
    ndarray
        A 3x3 rotation matrix.
    """
    # Compute the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Perform singular value decomposition
    V, S, W = np.linalg.svd(C)

    # Check if a reflection is needed
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        # Adjust the singular values and V matrix if reflection is needed
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Compute the rotation matrix
    U = np.dot(V, W)

    return U


def align_xyz(vec1, vec2, coord)->ndarray:
    """
    Align a set of coordinates by computing the rotation matrix using the Kabsch
    algorithm with two reference vectors and applying it to the coordinates.

    Parameters
    ----------
    vec1 : ndarray
        An array of shape (N, 3) representing the first reference vector.
    vec2 : ndarray
        An array of shape (N, 3) representing the second reference vector.
    coord : ndarray
        An array of shape (M, 3) representing the coordinates to be aligned.

    Returns
    -------
    ndarray
        The aligned coordinates.
    """
    # Compute the rotation matrix using the Kabsch algorithm
    rotmatrix = kabsch(vec1, vec2)

    # Apply the rotation matrix to the coordinates
    return np.dot(coord, rotmatrix)

from ase import Atoms
from ase.data import atomic_numbers
import numpy as np
from typing import Union, List

def from_ase_atoms(ase_obj: Union[Atoms, List[Atoms], List[dict]], name: str = "molecule") -> List[dict]:
    """
    Converts an ASE Atoms object, a list of Atoms objects, or a list of dictionaries with names and Atoms
    to a list of JSON-like dictionaries compatible with xyz_format_to_json.

    Parameters
    ----------
    ase_obj : ase.Atoms, List[ase.Atoms], or List[dict]
        - A single ASE Atoms object.
        - A list of ASE Atoms objects (names assigned as "1", "2", "3", ...).
        - A list of dictionaries, each with:
            - 'name': str, the molecule name.
            - 'atoms': ase.Atoms, the ASE Atoms object.
    name : str, optional
        The name of the molecule for a single Atoms object. Defaults to "molecule".
        Ignored for list inputs (uses indices or dictionary names).

    Returns
    -------
    xyz_jsons : List[dict]
        A list of dictionaries, each containing:
        - 'name': str, molecule name (from input, index, or default).
        - 'n_atoms': int, number of atoms.
        - 'coordinate': ndarray, array of shape (n_atoms, 5) with columns [atomic_number, x, y, z, index].

    Usage
    -----
    >>> from ase import Atoms
    >>> # Single molecule
    >>> atoms = Atoms(symbols=['H', 'H'], positions=[[0.0, 0.0, 0.7], [0.0, 0.0, 0.0]])
    >>> result = from_ase(atoms, name='H2')
    >>> print(result)
    [{'name': 'H2', 'n_atoms': 2, 'coordinate': array([[1., 0., 0., 0.7, 1.],
                                                       [1., 0., 0., 0.0, 2.]])}]
    >>>
    >>> # List of Atoms objects
    >>> atoms_list = [
    >>>     Atoms(symbols=['H', 'H'], positions=[[0.0, 0.0, 0.7], [0.0, 0.0, 0.0]]),
    >>>     Atoms(symbols=['O', 'H', 'H'], positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    >>> ]
    >>> result = from_ase(atoms_list)
    >>> print(result)
    [{'name': '1', 'n_atoms': 2, 'coordinate': array([[1., 0., 0., 0.7, 1.],
                                                      [1., 0., 0., 0.0, 2.]])},
     {'name': '2', 'n_atoms': 3, 'coordinate': array([[8.,  0.   ,  0.   ,  0.   ,  1.],
                                                      [1.,  0.757,  0.586,  0.   ,  2.],
                                                      [1., -0.757,  0.586,  0.   ,  3.]])}]
    >>>
    >>> # List of dictionaries
    >>> dict_list = [
    >>>     {'name': 'H2', 'atoms': Atoms(symbols=['H', 'H'], positions=[[0.0, 0.0, 0.7], [0.0, 0.0, 0.0]])},
    >>>     {'name': 'H2O', 'atoms': Atoms(symbols=['O', 'H', 'H'], positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])}
    >>> ]
    >>> result = from_ase(dict_list)
    >>> print(result)
    [{'name': 'H2', 'n_atoms': 2, 'coordinate': array([[1., 0., 0., 0.7, 1.],
                                                       [1., 0., 0., 0.0, 2.]])},
     {'name': 'H2O', 'n_atoms': 3, 'coordinate': array([[8.,  0.   ,  0.   ,  0.   ,  1.],
                                                        [1.,  0.757,  0.586,  0.   ,  2.],
                                                        [1., -0.757,  0.586,  0.   ,  3.]])}]
    """
    def _convert_single_atoms(atoms: Atoms, name: str, index: int = None) -> dict:
        # Get number of atoms
        n_atoms = len(atoms)
        
        # Get atomic numbers and positions
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        
        # Create index column (1-based indexing)
        indices = np.arange(1, n_atoms + 1).reshape(-1, 1)
        
        # Combine atomic numbers, positions, and indices into a single array
        coordinates = np.hstack([
            atomic_numbers.reshape(-1, 1),
            positions,
            indices
        ])
        
        # Create JSON-like dictionary
        xyz_json = {
            "name": name if name and index is None else str(index) if index is not None else "molecule",
            "n_atoms": n_atoms,
            "coordinate": coordinates.astype(float)
        }
        
        return xyz_json

    # Handle single Atoms object
    if isinstance(ase_obj, Atoms):
        return [_convert_single_atoms(ase_obj, name)]
    
    # Handle list of Atoms objects
    elif isinstance(ase_obj, (list, tuple)) and all(isinstance(item, Atoms) for item in ase_obj):
        return [
            _convert_single_atoms(atoms, "", idx + 1)
            for idx, atoms in enumerate(ase_obj)
        ]
    
    # Handle list of dictionaries
    elif isinstance(ase_obj, (list, tuple)) and all(
        isinstance(item, dict) and "name" in item and "atoms" in item and isinstance(item["atoms"], Atoms)
        for item in ase_obj
    ):
        return [
            _convert_single_atoms(item["atoms"], item["name"])
            for item in ase_obj
        ]
    
    else:
        raise TypeError(
            "ase_obj must be an ASE Atoms object, a list of Atoms objects, "
            "or a list of dictionaries with 'name' and 'atoms' keys"
        )

def xyz_format_to_json(xyz_coord:str|dict)->dict:
    """
    Converts an XYZ format coordinate (string, file path, or dictionary) to a JSON-like dictionary.

    The function takes input in the form of a string representing XYZ format coordinates, 
    a file path pointing to an XYZ format file, or a dictionary containing the molecule name 
    and coordinates in XYZ format. It then converts the input to a structured JSON-like 
    dictionary with the molecule name, number of atoms, and atomic coordinates.

    Parameters
    ----------
    xyz_coord : str or dict
        str:
            An XYZ format string or a file path to an XYZ format file.
        dict:
            A dictionary with the following keys:
            - "name": str
                The name of the molecule.
            - "coordinate": str
                The coordinates in XYZ format or a file path to an XYZ format file.

    Returns
    -------
    xyz_json : dict
        A JSON-like dictionary containing the parsed XYZ data with the following structure:
        - 'name': str
            Molecule name.
        - 'n_atoms': int
            Number of atoms.
        - 'coordinate': ndarray
            Array of atomic coordinates.

    Usage
    -----
    >>> molecule = {
    >>>     'name': 'aspirin',
    >>>     'coordinate': '''
    >>>     2
    >>>
    >>>     H 0.0 0.0 0.7
    >>>     H 0.0 0.0 0.0
    >>>     '''
    >>> }
    >>> xyz_json = xyz_format_to_json(molecule)
    """
    def _read_string(xyz:str)->str:
        """read xyz string or filepath
        """
        # xyz -> filepath
        if isfile(xyz):
            with open(xyz, "r") as file:
                xyz_string = file.read()
            name = basename(xyz)
            return name, xyz_string
        # xyz -> xyz format string
        else:
            name = ""
            return name, xyz

    if isinstance(xyz_coord, dict):
        name = xyz_coord["name"]
        _, xyz_string = _read_string(xyz_coord["coordinate"])

    if isinstance(xyz_coord, str):
        name, xyz_string = _read_string(xyz_coord)

    # number of atoms
    n_atoms = re.search(r"(\d+)", xyz_string).group(0)

    # split xyz string
    pattern = re.compile("([a-zA-Z]{1,2}(\s+-?\d+.\d+){3,3})+")
    xyz_lines = np.array(np.array(list(re.split(r'\s+', tup[0]) for tup in pattern.findall(xyz_string))))

    # converts atomic_symbol to atomic_number
    xyz_lines[:, 0] = np.array(list(element_symbol2atomic_number[symbol] for symbol in xyz_lines[:, 0]))

    # add atomic index
    index_col = np.arange(1, xyz_lines.shape[0] + 1, 1).reshape(-1, 1)
    xyz_lines = np.hstack([xyz_lines, index_col])

    # json format xyz
    xyz_json = {
        "name": name,
        "n_atoms": n_atoms,
        "coordinate": xyz_lines.astype(float)
    }

    return xyz_json



def open_xyz_files(xyz_coordinates:str|list)->list:
    """
    Opens and reads XYZ files, extracting headers and atomic coordinates, and converts them to a JSON-like format.

    This function can handle either a single trajectory file containing multiple XYZ frames or a list of XYZ file strings or dictionaries. 
    It extracts the relevant information from each XYZ file and converts it to a JSON-like format.

    Parameters
    ----------
    xyz_coordinates : str or list
        str:
            A file path to an XYZ trajectory file containing multiple frames.
        list:
            A list of str or dict. Refer to the docstring of `xyz_format_to_json`:
            - dict:
                - 'name': str, the name or identifier for the XYZ structure.
                - 'coordinate': str, the file path to an XYZ format file or the XYZ format string itself.
            - str:
                The file path or XYZ format string.

    Returns
    -------
    xyz_format_jsons : list
        A list of dictionaries in JSON-like format representing the parsed XYZ files. Each dictionary contains:
        - 'name': str, the name or identifier for the XYZ structure.
        - 'n_atoms': int, the number of atoms in the structure.
        - 'coordinate': ndarray, the atomic coordinates and atomic numbers.


    Usage
    -----
    >>> # From the list of xyz files
    >>> xyz_files = [
    >>>     {'name': 'reactant', 'coordinate': './sn2_reac.xyz'},
    >>>     {'name': 'TS', 'coordinate': './sn2_TS.xyz'},
    >>>     {'name': 'prod', 'coordinate': './sn2_prod.xyz'}
    >>> ]
    >>> xyz_format_jsons = open_xyz_files(xyz_files)
    >>>
    >>> # From a single traj file
    >>> xyz_format_jsons_from_traj = open_xyz_files('sn2_traj.xyz')
    """
    # traj file
    if isinstance(xyz_coordinates, str):
        with open(xyz_coordinates, "r") as file:
            traj_string = file.read()
        # find all xyz format strings
        pattern = re.compile("(\s?\d+\n.*\n(\s*[a-zA-Z]{1,2}(\s+-?\d+.\d+){3,3}\n?)+)")
        matched_xyz_formats = pattern.findall(traj_string)
        xyz_format_strings = list(tup[0] for tup in matched_xyz_formats)

        # convert xyz format stirng to json format
        traj_idx = list(idx+1 for idx in range(len(xyz_format_strings)))
        xyz_format_jsons = list(xyz_format_to_json({'name': idx, 'coordinate': xyz_string}) for idx, xyz_string in zip(traj_idx, xyz_format_strings))

        return xyz_format_jsons

    elif isinstance(xyz_coordinates, Iterable):
        return list(map(xyz_format_to_json, xyz_coordinates))

    else:
        raise TypeError("xyz_coordinates must be str or list")


def superimpose(xyz_format_jsons:list, option="aa", option_param:None|list=None)->list:
    """
    Superimpose multiple molecular structures based on various alignment options.

    Parameters
    ----------
    xyz_format_jsons : list
        A list of dictionaries in JSON-like format, each representing a molecule. Each dictionary contains:
        - "name": str, the name or identifier for the molecule.
        - "n_atoms": int, the number of atoms in the molecule.
        - "coordinate": ndarray, the atomic coordinates and atomic numbers.

    option : str, optional, default='aa'
        Alignment option. Supported options : ['aa', 'sa', 'a']
        - 'aa' : Align `all atoms`. Assumes that all molecules have atoms in the same order.
        - 'sa' : Align based on the `same atoms` across molecules using the indices provided in `option_param`.
        - 'a'  : Align specific `atoms` based on indices provided in `option_param`.

    option_param : list, optional, default=None
        Parameters specific to the selected alignment option.
        - Atomic index starts with 1.
        - For option='aa': None.
        - For option='sa': A list of atom indices. e.g. [1, 2, 3]
        - For option='a': A list of lists. e.g. [[1, 2, 3], [4, 5, 6]] # same order as xyz files

    Returns
    -------
    list
        A list of dictionaries which contains superimposed molecular cooridnate with JSON-like format.

    Usage
    -----
    >>> xyz_files = [
    >>>     {'name': 'reactant', 'coordinate': 'sn2_reac.xyz'},
    >>>     {'name': 'TS', 'coordinate': 'sn2_TS.xyz'},
    >>>     {'name': 'prod', 'coordinate': 'sn2_prod.xyz'}
    >>> ]
    >>> xyz_format_jsons = open_xyz_files(xyz_files)
    >>> superimposed_jsons = superimpose(xyz_format_jsons, option='sa', option_param=[1, 2, 5])
    """
    # copy xyz_format_jsons
    _xyz_format_jsons = deepcopy(xyz_format_jsons)

    # all atoms ( -aa option )
    if option=="aa":
        if option_param: print("\033[31m[WARNING]\033[0m", "`aa` option does not require `option_param`. `option_param` is ignored.")

        # `aa` option expects that each coordinates has the same order of atoms
        atomic_indice_list = list(map(lambda xyz_json : xyz_json.get("coordinate")[:, 0], _xyz_format_jsons))
        symbols_list = list(atomic_number2element_symbol[atomic_index[0]] for atomic_index in atomic_indice_list)
        # check each coordinates has same order
        if not np.all(np.array(symbols_list) == symbols_list[0]):
            raise ValueError("The `aa` option expects that each coordinates has the same order of atoms. \n Try other options like `sa` or `a`")

        # every molecule is overlaid on the first molecule
        first_molecule = _xyz_format_jsons[0]["coordinate"][:, 1:4]
        centroid = np.mean(first_molecule, axis=0)

        # center the first(reference) molecule
        first_molecule -= centroid
        _xyz_format_jsons[0]["coordinate"][:, 1:4] = first_molecule

        for mol_idx in range(len(_xyz_format_jsons)):
            # center the molecule
            centroid_mol = np.mean(_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4], axis=0)
            _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] -= centroid_mol

            # overlay each molecule on the first molecule
            _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] = align_xyz(
                vec1=_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4], # molecule to align
                vec2=first_molecule, # reference molecule
                coord=_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] # coordinates to align
              )

    # atoms ( -a option )
    elif option=="a":
        # `option_param` should be a list with the same size as the number of molecules to overlay
        if not option_param: raise ValueError("`a` option requires `option_param`")
        if not isinstance(option_param, Iterable): raise TypeError("`option_param` must be list")
        if not np.all(np.array(list(len(param) for param in option_param)) == len(option_param[0])): raise ValueError("all elements in `option_param` must have the same size")
        if not len(_xyz_format_jsons) == len(option_param): raise ValueError(f"""`a` option requires `option_param` to have the same length as `xyz_format_jsons`\n
         Number of molecules to overlay : {len(_xyz_format_jsons)}
         Length of option_param : {len(option_param)}""")
        if any(list((0 in param) for param in option_param)): raise ValueError("atomic indices start with 1, but 0 was found in `option_param`")

        # reset atomic indice
        option_param = list(
            list(param - 1 for param in mol_param) for mol_param in option_param
            )

        # every molecule is overlaid on the first molecule
        first_molecule_selected_atoms = _xyz_format_jsons[0]["coordinate"][:, 1:4][[option_param[0]]][0]
        centroid = np.mean(first_molecule_selected_atoms, axis=0)

        # center the first(reference) molecule
        _xyz_format_jsons[0]["coordinate"][:, 1:4] -= centroid

        for mol_idx in range(len(_xyz_format_jsons)):
            # center the molecule
            centroid_mol = np.mean(_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][option_param[mol_idx]], axis=0)
            _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] -= centroid_mol

            # overlay each molecule on the first molecule
            _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] = align_xyz(
                vec1=_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][[option_param[mol_idx]]][0], # molecule to align
                vec2=first_molecule_selected_atoms,
                coord=_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] # coordinates to align
              )

    # same atom ( -sa option )
    elif option=="sa":
        # `option_param` should be a list
        if not option_param: raise ValueError("`sa` option requires `option_param`")
        if not isinstance(option_param, Iterable): raise TypeError("`option_param` must be list")
        if 0 in option_param: raise ValueError("atomic indices start with 1, but 0 was found in `option_param`")

        # reset atomic indice
        option_param = list(param - 1 for param in option_param)

        # `sa` option expects that each coordinates has the same order of selected atoms
        selected_atomic_indice_list = list(map(lambda xyz_json : xyz_json.get("coordinate")[:, 0][[option_param]], _xyz_format_jsons))
        selected_symbols_list = list(atomic_number2element_symbol[atomic_number[0][0]] for atomic_number in selected_atomic_indice_list)

        # check each coordinates has same order
        if not np.all(np.array(selected_symbols_list) == selected_symbols_list[0]):
            raise ValueError("The `aa` option expects that each coordinates has the same order of atoms. \n Try other options like `sa` or `a`")

        # every molecule is overlaid on the first molecule
        first_molecule_selected_atoms = _xyz_format_jsons[0]["coordinate"][:, 1:4][[option_param]][0]
        centroid = np.mean(first_molecule_selected_atoms, axis=0)

        # center the first(reference) molecule
        _xyz_format_jsons[0]["coordinate"][:, 1:4] -= centroid

        for mol_idx in range(len(_xyz_format_jsons)):
            # center the molecule
            #centroid_mol = np.mean(_xyz_format_jsons[mol_idx]["coordinate"][:, 1:][[option_param]], axis=0)
            centroid_mol = np.mean(_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][option_param], axis=0)
            _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] -= centroid_mol
            # overlay each molecule on the first molecule
            _xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] = align_xyz(
                vec1=_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4][[option_param]][0], # molecule to align
                vec2=first_molecule_selected_atoms,
                coord=_xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] # coordinates to align
              )
    elif option==None:
        return xyz_format_jsons

    else:
        raise ValueError(f"Unsupported option : {option}")

    return _xyz_format_jsons


def xyz2molecular_graph(xyz_format_jsons:list, covalent_radius_percent:float=108.):
    """
    Get molecular graph connectivity information like adjacency matrix & bond length table.

    Parameters
    ----------
    xyz_format_jsons : list
        A list of dictionaries, each representing a molecular coordinate in JSON format. Each dictionary contains:
        - 'name': str, the name or identifier for the molecule.
        - 'n_atoms': int, the number of atoms in the molecule.
        - 'coordinate': ndarray, the atomic coordinates and atomic numbers.

    covalent_radius_percent : float, optional, default=108.0
        The percentage of the standard covalent radii to use for determining bond distances.

    Returns
    -------
    None
        The function updates the `xyz_format_jsons` list in place by adding:
        - 'adjacency_matrix': ndarray, the adjacency matrix representing bonds between atoms.
        - 'bond_length_table': ndarray, a table of bond lengths with columns ['atom_1_idx', 'atom_2_idx', 'distance'].

    Notes
    -----
    - Bond lengths are determined based on covalent radii adjusted by `covalent_radius_percent`.
    """
    def _covalent_radii(element:str, percent:float):
        """resize covalent radius
        """
        radius = covalent_radii[element]
        radius *= (percent / 100)
        return radius

    # get molecular connetivity & bond length
    for mol_idx in range(len(xyz_format_jsons)):
        # get interatomic distance(L2 norm) matrix
        atomic_coordinates = xyz_format_jsons[mol_idx]["coordinate"][:, 1:4] # (N, 3)
        L2_matrix = squareform(pdist(atomic_coordinates, 'euclidean')) # (N, N)

        # get sum of atomic radii matrix
        symbols_vector = np.array(list(
            atomic_number2element_symbol[atomic_number] for atomic_number in xyz_format_jsons[mol_idx]["coordinate"][:, 0]
              )) # (N, 3)
        radii_vector = np.array(list(_covalent_radii(symbol, covalent_radius_percent) for symbol in symbols_vector)) # (N, 3)
        radii_sum_matrix = np.add.outer(radii_vector, radii_vector) # (N, N)

        # get adjacency(bond) matrix
        adjacency_matrix = np.array(L2_matrix <= radii_sum_matrix) # (N, N)
        np.fill_diagonal(adjacency_matrix, 0) # diagonal means self-bonding
        xyz_format_jsons[mol_idx]["adjacency_matrix"] = adjacency_matrix

        # bond length matrix = adjacency_matrix * L2_matrix
        bond_length_matrix = adjacency_matrix * L2_matrix # (N, N)

        # get bond length table
        # remove duplicated values. Rba = Rab ( symmetrix mat. )
        bond_length_matrix[np.triu_indices_from(bond_length_matrix, k=1)] = 0
        mask = ~np.equal(bond_length_matrix, 0)
        # bond ( atom_pair ) & bond length
        atom_pairs = np.array(np.nonzero(mask)).T
        length = bond_length_matrix[mask]
        # bond length table | atom_1_idx | "atom_2_idx | distance | 
        # idx start with 1
        #atom_1 = symbols_vector[atom_pairs[:, 0]]
        #atom_2 = symbols_vector[atom_pairs[:, 1]]
        #bond_length_table = np.column_stack((atom_1, atom_2, length))
        bond_length_table = np.column_stack((atom_pairs[:, 0] + 1, atom_pairs[:, 1] + 1, length))
        xyz_format_jsons[mol_idx]["bond_length_table"] = bond_length_table
