from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts
from graph_util import *

np.random.seed(123)


def extract_graph(data_path, out_file_path, max_atom_num, label_name=None):
    import os
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    data_pd = pd.read_csv(data_path)
    smiles_list = data_pd['SMILES'].tolist()

    symbol_candidates = set()
    atom_attribute_dim = num_atom_features()
    bond_attribute_dim = num_bond_features()

    node_attribute_matrix_list = []
    bond_attribute_matrix_list = []
    adjacent_matrix_list = []
    distance_matrix_list = []
    valid_index = []

    ###
    degree_set = set()
    h_num_set = set()
    implicit_valence_set = set()
    charge_set = set()
    ###

    for line_idx, smiles in enumerate(smiles_list):
        smiles = smiles.strip()
        mol = MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformers()[0]
        feats = factory.GetFeaturesForMol(mol)
        acceptor_atom_ids = map(lambda x: x.GetAtomIds()[0], filter(lambda x: x.GetFamily() =='Acceptor', feats))
        donor_atom_ids = map(lambda x: x.GetAtomIds()[0], filter(lambda x: x.GetFamily() =='Donor', feats))

        adjacent_matrix = np.zeros((max_atom_num, max_atom_num))
        adjacent_matrix = adjacent_matrix.astype(int)
        distance_matrix = np.zeros((max_atom_num, max_atom_num))
        node_attribute_matrix = np.zeros((max_atom_num, atom_attribute_dim))
        node_attribute_matrix = node_attribute_matrix.astype(int)

        if len(mol.GetAtoms()) > max_atom_num:
            print('Outlier {} has {} atoms'.format(line_idx, mol.GetNumAtoms()))
            continue
        valid_index.append(line_idx)

        atom_positions = [None for _ in range(mol.GetNumAtoms()+1)]
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol_candidates.add(atom.GetSymbol())
            atom_positions[atom_idx] = conformer.GetAtomPosition(atom_idx)
            degree_set.add(atom.GetDegree())
            h_num_set.add(atom.GetTotalNumHs())
            implicit_valence_set.add(atom.GetImplicitValence())
            charge_set.add(atom.GetFormalCharge())
            node_attribute_matrix[atom_idx] = extract_atom_features(atom,
                                                                    is_acceptor=atom_idx in acceptor_atom_ids,
                                                                    is_donor=atom_idx in donor_atom_ids)
        node_attribute_matrix_list.append(node_attribute_matrix)

        for idx_i in range(mol.GetNumAtoms()):
            for idx_j in range(idx_i+1, mol.GetNumAtoms()):
                distance = get_atom_distance(conformer.GetAtomPosition(idx_i),
                                             conformer.GetAtomPosition(idx_j))
                distance_matrix[idx_i, idx_j] = distance
                distance_matrix[idx_j, idx_i] = distance
        distance_matrix_list.append(distance_matrix)

        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_index = begin_atom.GetIdx()
            end_index = end_atom.GetIdx()
            adjacent_matrix[begin_index, end_index] = 1
            adjacent_matrix[end_index, begin_index] = 1
        adjacent_matrix_list.append(adjacent_matrix)

    adjacent_matrix_list = np.asarray(adjacent_matrix_list)
    distance_matrix_list = np.asarray(distance_matrix_list)
    node_attribute_matrix_list = np.asarray(node_attribute_matrix_list)
    bond_attribute_matrix_list = np.asarray(bond_attribute_matrix_list)
    print('adjacent matrix shape\t', adjacent_matrix_list.shape)
    print('distance matrix shape\t', distance_matrix_list.shape)
    print('node attr matrix shape\t', node_attribute_matrix_list.shape)
    print('bond attr matrix shape\t', bond_attribute_matrix_list.shape)
    print(symbol_candidates)
    print('{} valid out of {}'.format(len(valid_index), len(smiles_list)))

    print('degree set:\t', degree_set)
    print('h num set: \t', h_num_set)
    print('implicit valence set: \t', implicit_valence_set)
    print('charge set:\t', charge_set)

    if label_name is None:
        np.savez_compressed(out_file_path,
                            adjacent_matrix_list=adjacent_matrix_list,
                            distance_matrix_list=distance_matrix_list,
                            node_attribute_matrix_list=node_attribute_matrix_list,
                            bond_attribute_matrix_list=bond_attribute_matrix_list)
    else:
        true_labels = data_pd[label_name].tolist()
        true_labels = np.array(true_labels)
        valid_index = np.array(valid_index)
        true_labels = true_labels[valid_index]
        np.savez_compressed(out_file_path,
                            adjacent_matrix_list=adjacent_matrix_list,
                            distance_matrix_list=distance_matrix_list,
                            node_attribute_matrix_list=node_attribute_matrix_list,
                            bond_attribute_matrix_list=bond_attribute_matrix_list,
                            label_name=true_labels)
    print()
    return


def extract_graph_multi_tasks_SDF(data_path, sdf_data_path, out_file_path, max_atom_num, task_list, clean_mols=False):
    data_pd = pd.read_csv(data_path)
    smiles_list = data_pd['SMILES'].tolist()
    suppl = Chem.SDMolSupplier(sdf_data_path, clean_mols, False, False)

    symbol_candidates = set()
    atom_attribute_dim = num_atom_features(explicit_H=True)
    bond_attribute_dim = num_bond_features()

    node_attribute_matrix_list = []
    bond_attribute_matrix_list = []
    adjacent_matrix_list = []
    distance_matrix_list = []
    valid_index = []

    ###
    degree_set = set()
    implicit_valence_set = set()
    charge_set = set()
    hybridization_set = set()
    ###

    for line_idx, mol in enumerate(suppl):
        conformer = mol.GetConformers()[0]

        adjacent_matrix = np.zeros((max_atom_num, max_atom_num))
        adjacent_matrix = adjacent_matrix.astype(int)
        distance_matrix = np.zeros((max_atom_num, max_atom_num))
        node_attribute_matrix = np.zeros((max_atom_num, atom_attribute_dim))
        node_attribute_matrix = node_attribute_matrix.astype(int)

        if len(mol.GetAtoms()) > max_atom_num:
            print('Outlier {} has {} atoms'.format(line_idx, mol.GetNumAtoms()))
            continue
        valid_index.append(line_idx)

        atom_positions = [None for _ in range(mol.GetNumAtoms()+1)]
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol_candidates.add(atom.GetSymbol())
            atom_positions[atom_idx] = conformer.GetAtomPosition(atom_idx)
            ####
            degree_set.add(atom.GetDegree())
            charge_set.add(atom.GetFormalCharge())
            implicit_valence_set.add(atom.GetImplicitValence())
            hybridization_set.add(atom.GetHybridization())
            ####
            node_attribute_matrix[atom_idx] = extract_atom_features(atom, explicit_H=True)
        node_attribute_matrix_list.append(node_attribute_matrix)

        for idx_i in range(mol.GetNumAtoms()):
            for idx_j in range(idx_i+1, mol.GetNumAtoms()):
                distance = get_atom_distance(conformer.GetAtomPosition(idx_i),
                                             conformer.GetAtomPosition(idx_j))
                distance_matrix[idx_i, idx_j] = distance
                distance_matrix[idx_j, idx_i] = distance
        distance_matrix_list.append(distance_matrix)

        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_index = begin_atom.GetIdx()
            end_index = end_atom.GetIdx()
            adjacent_matrix[begin_index, end_index] = 1
            adjacent_matrix[end_index, begin_index] = 1
        adjacent_matrix_list.append(adjacent_matrix)

    adjacent_matrix_list = np.asarray(adjacent_matrix_list)
    distance_matrix_list = np.asarray(distance_matrix_list)
    node_attribute_matrix_list = np.asarray(node_attribute_matrix_list)
    bond_attribute_matrix_list = np.asarray(bond_attribute_matrix_list)
    print('adjacent matrix shape\t', adjacent_matrix_list.shape)
    print('distance matrix shape\t', distance_matrix_list.shape)
    print('node attr matrix shape\t', node_attribute_matrix_list.shape)
    print('bond attr matrix shape\t', bond_attribute_matrix_list.shape)
    print(symbol_candidates)
    print('{} valid out of {}'.format(len(valid_index), len(smiles_list)))

    print('degree set:\t', degree_set)
    print('implicit valence set: \t', implicit_valence_set)
    print('charge set:\t', charge_set)
    print('hybridization set:\t', hybridization_set)

    kwargs = {}
    kwargs['adjacent_matrix_list'] = adjacent_matrix_list
    kwargs['distance_matrix_list'] = distance_matrix_list
    kwargs['node_attribute_matrix_list'] = node_attribute_matrix_list
    kwargs['bond_attribute_matrix_list'] = bond_attribute_matrix_list
    for task in task_list:
        true_labels = data_pd[task].tolist()
        true_labels = np.array(true_labels)
        valid_index = np.array(valid_index)
        true_labels = true_labels[valid_index]
        kwargs[task] = true_labels
    np.savez_compressed(out_file_path, **kwargs)
    print()
    return
