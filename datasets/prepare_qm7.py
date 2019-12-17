from __future__ import print_function

import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.model_selection import KFold
import os
import sys
from data_preprocess import *


np.random.seed(123)
max_atom_num = 55
K = 5


def prepare_fingerprints(target_name='qm7'):
    whole_data_pd = pd.read_csv('{}.csv'.format(target_name))

    column = ['smiles', 'u0_atom']
    data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
    data_pd.columns = ['SMILES', target_name]
    print(data_pd.columns)

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
    data_pd = data_pd[['SMILES', 'Fingerprints', target_name]]

    if not os.path.exists(target_name):
        os.makedirs(target_name)
    print('total shape\t', data_pd.shape)

    kf = KFold(n_splits=K, shuffle=True)
    for i, (_,index) in enumerate(kf.split(np.arange(data_pd.shape[0]))):
        temp_pd = data_pd.iloc[index]
        print(i, '\t', temp_pd.shape)
        print(index)
        temp_pd.to_csv('{}/{}.csv.gz'.format(target_name, i), compression='gzip', index=None)
    return


if __name__ == '__main__':
    target_name = 'qm7'
    prepare_fingerprints(target_name)
    print()

    for i in range(K):
        extract_graph(data_path='{}/{}.csv.gz'.format(target_name, i),
                      out_file_path='{}/{}_graph.npz'.format(target_name, i),
                      label_name='qm7',
                      max_atom_num=max_atom_num)
