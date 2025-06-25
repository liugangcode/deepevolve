import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# Suppress RDKit error messages globally
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
from torch_geometric.data import Data
import torch


def convert_to_pytorch_data(X, y=None):
    """Convert numpy arrays to PyTorch Geometric data format."""
    pyg_graph_list = []
    for idx, smiles_or_mol in enumerate(X):
        if y is not None:
            properties = y[idx]
        else:
            properties = None
        graph = graph_from_smiles(smiles_or_mol, properties)
        g = Data()
        g.num_nodes = graph["num_nodes"]
        g.edge_index = torch.from_numpy(graph["edge_index"])

        del graph["num_nodes"]
        del graph["edge_index"]

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.from_numpy(graph["edge_feat"])
            del graph["edge_feat"]

        if graph["node_feat"] is not None:
            g.x = torch.from_numpy(graph["node_feat"])
            del graph["node_feat"]

        if graph["y"] is not None:
            g.y = torch.from_numpy(graph["y"])
            del graph["y"]

        pyg_graph_list.append(g)

    return pyg_graph_list


def graph_from_smiles(smiles_or_mol, properties):
    """
    Converts SMILES string or RDKit molecule to graph Data object

    Parameters
    ----------
    smiles_or_mol : Union[str, rdkit.Chem.rdchem.Mol]
        SMILES string or RDKit molecule object
    properties : Any
        Properties to include in the graph

    Returns
    -------
    dict
        Graph object dictionary
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))

    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    # Handle properties and augmented properties
    props_list = []
    if properties is not None:
        props_list.append(np.array(properties, dtype=np.float32))
    if props_list:
        combined_props = np.concatenate(props_list)
        graph["y"] = combined_props.reshape(1, -1)
    else:
        graph["y"] = np.full((1, 1), np.nan, dtype=np.float32)

    return graph


# allowable multiple choice node and edge features
allowable_features = {
    # atom types: 1-118, 119 is masked atom, 120 is misc (e.g. * for polymers)
    # index: 0-117, 118, 119
    "possible_atomic_num_list": list(range(1, 120)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "misc",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        safe_index(
            allowable_features["possible_chirality_list"], str(atom.GetChiralTag())
        ),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def get_atom_feature_dims():
    return list(
        map(
            len,
            [
                allowable_features["possible_atomic_num_list"],
                allowable_features["possible_chirality_list"],
                allowable_features["possible_degree_list"],
                allowable_features["possible_formal_charge_list"],
                allowable_features["possible_numH_list"],
                allowable_features["possible_number_radical_e_list"],
                allowable_features["possible_hybridization_list"],
                allowable_features["possible_is_aromatic_list"],
                allowable_features["possible_is_in_ring_list"],
            ],
        )
    )


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_bond_feature_dims():
    return list(
        map(
            len,
            [
                allowable_features["possible_bond_type_list"],
                allowable_features["possible_bond_stereo_list"],
                allowable_features["possible_is_conjugated_list"],
            ],
        )
    )


def atom_feature_vector_to_dict(atom_feature):
    [
        atomic_num_idx,
        chirality_idx,
        degree_idx,
        formal_charge_idx,
        num_h_idx,
        number_radical_e_idx,
        hybridization_idx,
        is_aromatic_idx,
        is_in_ring_idx,
    ] = atom_feature

    feature_dict = {
        "atomic_num": allowable_features["possible_atomic_num_list"][atomic_num_idx],
        "chirality": allowable_features["possible_chirality_list"][chirality_idx],
        "degree": allowable_features["possible_degree_list"][degree_idx],
        "formal_charge": allowable_features["possible_formal_charge_list"][
            formal_charge_idx
        ],
        "num_h": allowable_features["possible_numH_list"][num_h_idx],
        "num_rad_e": allowable_features["possible_number_radical_e_list"][
            number_radical_e_idx
        ],
        "hybridization": allowable_features["possible_hybridization_list"][
            hybridization_idx
        ],
        "is_aromatic": allowable_features["possible_is_aromatic_list"][is_aromatic_idx],
        "is_in_ring": allowable_features["possible_is_in_ring_list"][is_in_ring_idx],
    }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx, bond_stereo_idx, is_conjugated_idx] = bond_feature

    feature_dict = {
        "bond_type": allowable_features["possible_bond_type_list"][bond_type_idx],
        "bond_stereo": allowable_features["possible_bond_stereo_list"][bond_stereo_idx],
        "is_conjugated": allowable_features["possible_is_conjugated_list"][
            is_conjugated_idx
        ],
    }

    return feature_dict


def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))


def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]
