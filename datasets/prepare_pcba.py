from __future__ import print_function

import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts
from sklearn.model_selection import KFold
import os
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../graph_methods')
from graph_util import *
import nltk
import csv
from data_preprocess import *
from prepare_n_grammed_graph import *

np.random.seed(123)
target_names = [
    'PCBA-927', 'PCBA-903', 'PCBA-493208', 'PCBA-904', 'PCBA-885',
    'PCBA-914', 'PCBA-915', 'PCBA-720532', 'PCBA-883', 'PCBA-891', 'PCBA-899', 'PCBA-884'
]

max_atom_num = 100
K = 5


def prepare_fingerprints_PCBA(dataset_name):
    whole_data_pd = pd.read_csv('{}.csv.gz'.format(dataset_name))

    for target_name in target_names:
        print(target_name)
        column = [target_name, 'mol_id', 'smiles']
        data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
        data_pd = data_pd.rename(columns={"smiles": "SMILES", "mol_id": "Molecule"})

        morgan_fps = []
        valid_index = []

        index_list = data_pd.index.tolist()
        smiles_list = data_pd['SMILES'].tolist()
        for idx, smiles in zip(index_list, smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if len(mol.GetAtoms()) > max_atom_num:
                print('Outlier {} has {} atoms'.format(idx, mol.GetNumAtoms()))
                continue
            valid_index.append(idx)
            fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            morgan_fps.append(fingerprints.ToBitString())

        data_pd = data_pd.ix[valid_index]
        data_pd['Fingerprints'] = morgan_fps
        data_pd = data_pd[['Molecule', 'SMILES', 'Fingerprints', target_name]]

        y_label = data_pd[target_name].tolist()
        y_label = np.array(y_label)

        directory = '{}/{}'.format(dataset_name, target_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        print('total shape\t', data_pd.shape)
        skf = StratifiedKFold(n_splits=K, shuffle=True)
        for i, (_, index) in enumerate(skf.split(y_label, y_label)):
            temp_pd = data_pd.iloc[index]
            print(i, '\t', temp_pd.shape)
            temp_pd.to_csv('{}/{}.csv.gz'.format(directory, i), compression='gzip', index=None)
    return


def get_hit_ratio():
    for target_name in target_names:
        y_label = []
        for i in range(1):
            data_path = 'pcba/{}/{}_graph.npz'.format(target_name, i)
            data = np.load(data_path)
            y_label.extend(data['label_name'])
        y_label = np.stack(y_label)
        non_missing_idx = np.where((np.isnan(y_label)==False))
        y_label = y_label[non_missing_idx]
        hit_ratio = 1.0 * sum(y_label) / len(y_label)
        print('\'{}\': {},'.format(target_name, hit_ratio))


if __name__ == '__main__':
    dataset_name = 'pcba'
    prepare_fingerprints_PCBA(dataset_name)

    for target_name in target_names:
        directory = '{}/{}'.format(dataset_name, target_name)
        for i in range(K):
            extract_graph(data_path='{}/{}.csv.gz'.format(directory, i),
                          out_file_path='{}/{}_graph.npz'.format(directory, i),
                          label_name=target_name,
                          max_atom_num=max_atom_num)

    get_hit_ratio()
