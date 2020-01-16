from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts
from sklearn.model_selection import StratifiedKFold
from data_preprocess import *
import os

np.random.seed(123)
max_atom_num = 100
K = 5


def prepare_fingerprints_HIV(dataset_name):
    whole_data_pd = pd.read_csv('{}.csv'.format(dataset_name))
    print(whole_data_pd.shape, '\t', whole_data_pd.columns)

    column = ['HIV_active', 'smiles']
    data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
    data_pd = data_pd.rename(columns={'smiles': 'SMILES', 'HIV_active': dataset_name})

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
    data_pd = data_pd[['SMILES', 'Fingerprints', dataset_name]]

    y_label = data_pd[dataset_name].tolist()
    y_label = np.array(y_label)

    directory = '{}'.format(dataset_name)
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
    directory = '{}'.format(dataset_name)
    y_label = []
    for i in range(4):
        data_path = '{}/{}_graph.npz'.format(directory, i)
        data = np.load(data_path)
        y_label.extend(data['label_name'])
    y_label = np.stack(y_label)
    hit_ratio = 1.0 * sum(y_label) / len(y_label)
    print('\'{}\': {},'.format(dataset_name, hit_ratio))


if __name__ == '__main__':
    dataset_name = 'hiv'
    prepare_fingerprints_HIV(dataset_name)

    directory = '{}'.format(dataset_name)
    for i in range(K):
        extract_graph(data_path='{}/{}.csv.gz'.format(directory, i),
                      out_file_path='{}/{}_graph.npz'.format(directory, i),
                      label_name=dataset_name,
                      max_atom_num=max_atom_num)

    get_hit_ratio()