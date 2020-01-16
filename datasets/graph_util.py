from __future__ import print_function

import math
import numpy as np
from rdkit import Chem
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

atom_candidates = ['C', 'Cl', 'I', 'F', 'O', 'N', 'P', 'S', 'Br', 'Unknown']

possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: 1 if x == s else 0, allowable_set)


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        logging.info('Unknown detected: {}'.format(x))
        x = allowable_set[-1]
    return map(lambda s: 1 if x == s else 0, allowable_set)


def extract_atom_features(atom, explicit_H=False, is_acceptor=0, is_donor=0):
    if explicit_H:
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(), atom_candidates) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3]) +
                        one_of_k_encoding(atom.GetIsAromatic(), [0, 1])
                        )
    else:
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(), atom_candidates) +
                        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                        one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3]) +
                        one_of_k_encoding(atom.GetIsAromatic(), [0, 1]) +
                        one_of_k_encoding(is_acceptor, [0, 1]) +
                        one_of_k_encoding(is_donor, [0, 1])
                        )


def extract_bond_features(bond):
    bt = bond.GetBondType()
    bond_features = np.array([bt == Chem.rdchem.BondType.SINGLE,
                              bt == Chem.rdchem.BondType.DOUBLE,
                              bt == Chem.rdchem.BondType.TRIPLE,
                              bt == Chem.rdchem.BondType.AROMATIC,
                              bond.GetIsConjugated(),
                              bond.IsInRing()])
    bond_features = bond_features.astype(int)
    return bond_features


def num_atom_features(explicit_H=False):
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(extract_atom_features(a, explicit_H))


def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(extract_bond_features(simple_mol.GetBonds()[0]))


def get_atom_distance(atom_a, atom_b):
    return math.sqrt((atom_a.x - atom_b.x) ** 2 + (atom_a.y - atom_b.y) ** 2 + (atom_a.z - atom_b.z) ** 2)