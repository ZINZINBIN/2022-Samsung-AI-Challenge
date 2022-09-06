import numpy as np
import pandas as pd
import torch
from typing import List, Union, Literal, Optional
from mendeleev.fetch import fetch_ionization_energies, fetch_table
from tqdm.auto import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler

# total atom list observed in train / test data
# ATOMS_LIST = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
# ATOMS_NUM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
# ATOMS_DEGREE = [0, 1, 2, 3, 4, 5, 6]
# ATOMS_HYBRID = [2,3,4,5,6]

# atom properties 
ATOMS_LIST = ['C','N','O','F','S','Cl','Br']
ATOMS_NUM_LIST = [6, 7, 8, 9, 16, 17, 35]
ATOMS_DEGREE = [1, 2, 3, 4]
ATOMS_NUMHS = [0, 1, 2, 3]
ATOMS_VALENCE = [0, 1, 2, 3]
ATOMS_AROMATIC = [0, 1]
ATOMS_RING = [0,1]
ATOMS_HYBRID = [2,3,4]

# additional
atom_properties = [
    'atomic_weight', 'atomic_radius', 'atomic_volume', 'electron_affinity',
    'dipole_polarizability', 'vdw_radius', 'en_pauling'
]

# bond properties
BOND_TYPE = {'SINGLE': 0,'DOUBLE': 1,'TRIPLE': 2,'AROMATIC': 3}
BOND_AROMATIC = [0,1]
BOND_CONJUGATED = [0,1]
BOND_RING = [0,1]

# molecular properties
MOL_PROPERTIES = [
    'MolMR', 'NHOHCount', 'NOCount',
    'NumHAcceptors', 'NumHDonors',
    'NumHeteroatoms', 'NumValenceElectrons',
    'MaxPartialCharge', 'MinPartialCharge',
    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'NumAromaticHeterocycles', 'NumAromaticCarbocycles',
    'NumSaturatedHeterocycles', 'NumSaturatedCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticCarbocycles',
    'RingCount', 'FractionCSP3', 'TPSA', 'LabuteASA'
]

# generate fingerprint from dataset
def get_fingerprint(df:pd.DataFrame):
    fp_numpy_list = []
    for idx, row in tqdm(df.iterrows(), desc = "Extract Fingerprint", total = len(df)):
        mol = Chem.MolFromSmiles(row["SMILES"])
        fp = Chem.RDKFingerprint(mol)
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_numpy_list.append(arr)
    df['fp'] = fp_numpy_list
    return df

# extract the total atoms observed in dataset
def get_atom_list(df : pd.DataFrame):
    atoms_list = [] 
    for smiles in tqdm(df['SMILES']):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = []
        for idx, atom in enumerate(mol.GetAtoms()):
            atoms.append(atom.GetSymbol())
        
        atoms = np.unique(atoms).tolist()
        atoms_list.extend(atoms)

    atoms_list = np.unique(atoms_list)
    print("total atom : ", len(atoms_list))
    return atoms_list.tolist()

def get_atom_table():
    atoms = np.array(ATOMS_NUM_LIST)
    features = atom_properties

    elem_df = fetch_table('elements')
    feature_df = elem_df.loc[atoms-1, features]
    feature_df.set_index(atoms, inplace=True)
    ies = fetch_ionization_energies()
    final_df = pd.concat([feature_df, ies.loc[atoms]], axis=1)
    scaled_df = final_df.copy()
    scaled_df.iloc[:] = scale(final_df.iloc[:])
    return scaled_df

# generate new dataframe with molecular properties
def get_mol_properties(df : pd.DataFrame, mol_properties : List = MOL_PROPERTIES):
    
    for idx, mol in tqdm(enumerate(df.loc[:, 'SMILES']), desc='Molecular Feature', total=len(df)):
        mol = Chem.AddHs(Chem.MolFromSmiles(mol))
        for properties in mol_properties:
            df.loc[idx, properties] = getattr(Descriptors, properties)(mol)

    df.replace(np.inf, 0, inplace=True)
    df = df.fillna(0)
    scaled_df = df.copy()
    scaled_df.loc[:, mol_properties] = scale(df.loc[:, mol_properties])
    return scaled_df

def char2idx(x : str, allowable_set : List)->List[int]:
    if x not in allowable_set:
        return [len(allowable_set)] 
    return [allowable_set.index(x)]

atom_table = get_atom_table()

def atom_feature(atom, table:pd.DataFrame = atom_table):
    features = []

    # Embedding Features : topological info
    features.extend(char2idx(atom.GetSymbol(), ATOMS_LIST))
    features.extend(char2idx(atom.GetDegree(), ATOMS_DEGREE))
    features.extend(char2idx(atom.GetTotalNumHs(), ATOMS_NUMHS))
    features.extend(char2idx(atom.GetImplicitValence(), ATOMS_VALENCE))
    features.extend(char2idx(int(atom.GetIsAromatic()), ATOMS_AROMATIC))
    features.extend(char2idx(int(atom.IsInRing()), ATOMS_RING))
    features.extend(char2idx(atom.GetHybridization(), ATOMS_HYBRID))

    # Continuous Features : chemical info
    # features += list(table.loc[atom.GetAtomicNum()].values)

    return np.array(features)

def edge_feature(bond):

    bond_type = bond.GetBondType().name
    if bond_type not in BOND_TYPE.keys():
        bond2idx = len(list(BOND_TYPE.items()))
    else:
        bond2idx = int(BOND_TYPE[bond_type])

    features = []
    features.extend([bond2idx])
    features.extend(char2idx(int(bond.GetIsAromatic()), BOND_AROMATIC))
    features.extend(char2idx(int(bond.GetIsConjugated()), BOND_CONJUGATED))
    features.extend(char2idx(int(bond.IsInRing()), BOND_RING))

    return np.array(features)

def convert_data_from_smiles(row, idx : int, mode : Literal['train', 'submission'], pred_col : Optional[Literal['Reorg_g','Reorg_ex', 'Multi']] = None):

    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    adj = Chem.GetAdjacencyMatrix(mol)

    features = []

    for _, atom in enumerate(mol.GetAtoms()):
        features.append(atom_feature(atom))

    bonds = []

    for idx_i in range(mol.GetNumAtoms()):
        for idx_j in range(mol.GetNumAtoms()):
            if adj[idx_i,idx_j] == 1:
                bonds.append([idx_i, idx_j])

    bonds_attr = []
    
    for atom in mol.GetAtoms():
        for bond in atom.GetBonds():
            bonds_attr.append(edge_feature(bond))
   
    features = torch.from_numpy(np.array(features)).float()
    bonds = torch.from_numpy(np.array(bonds)).long().t().contiguous()
    bonds_attr = torch.from_numpy(np.array(bonds_attr)).float().contiguous()

    if mode == 'train':
        if pred_col == 'Multi':
            pred_col = ['Reorg_g','Reorg_ex']
        pred = row[pred_col]
        pred = torch.tensor([pred], dtype = torch.float)
        return Data(x = features, edge_index = bonds, edge_attr = bonds_attr, y = pred, idx = idx)
    else:
        return Data(x = features, edge_index = bonds, edge_attr = bonds_attr, y = None, idx = idx)
    
def generate_dataset(df : pd.DataFrame, mode : Literal['train', 'submission'], pred_col : Optional[Literal['Reorg_g','Reorg_ex', 'Multi']] = None):
    dataset = []
    for idx, row in tqdm(df.iterrows()):
        data = convert_data_from_smiles(row, idx, mode, pred_col)
        dataset.append(data)
    return dataset

def generate_dataloader(
    df : pd.DataFrame, 
    mode : Literal['train', 'submission'] = 'train', 
    test_size : Optional[float] = None, 
    valid_size : float = 0.2,
    batch_size : int = 128, 
    pred_col : Optional[Literal['Reorg_g', 'Reorg_ex', 'Multi']] = 'Reorg_g'
    ):

    dataset = generate_dataset(df, mode, pred_col)
    indices = range(0, len(dataset))

    if mode == 'train' and test_size is not None:
        train_indices, test_indices = train_test_split(indices, test_size = test_size, random_state = 42, shuffle = True)
        train_indices, valid_indices = train_test_split(train_indices, test_size = valid_size, random_state = 42, shuffle = True)

        train_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(train_indices))
        valid_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(valid_indices))
        test_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(test_indices))

        return train_loader, valid_loader, test_loader

    elif mode == 'train' and test_size is None:
        test_indices = None
        train_indices, valid_indices = train_test_split(indices, test_size = valid_size, random_state = 42, shuffle = True)
        train_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(train_indices))
        valid_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(valid_indices))

        return train_loader, valid_loader, None

    else:
        submission_loader = DataLoader(dataset, batch_size, shuffle = False)
        return submission_loader

def generate_dataloader_cv(
    df : Optional[pd.DataFrame] = None, 
    mode : Literal['train', 'submission'] = 'train', 
    batch_size : int = 128, 
    train_indices : Optional[List] = None,
    valid_indices : Optional[List] = None,
    pred_col : Optional[Literal['Reorg_g', 'Reorg_ex', 'Multi']] = 'Reorg_g'
    ):

    dataset = generate_dataset(df, mode, pred_col)
    train_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(train_indices))
    valid_loader = DataLoader(dataset, batch_size, sampler = SubsetRandomSampler(valid_indices))
    return train_loader, valid_loader


def extract_degree(dataset : Dataset)->torch.Tensor:
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg