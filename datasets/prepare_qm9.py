from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.model_selection import KFold
from data_preprocess import *
import os


np.random.seed(123)
max_atom_num = 55
K = 5
qm9_tasks = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
             'u298_atom', 'h298_atom', 'g298_atom']


def prepare(dataset_name, clean_mols=False):
    whole_data_pd = pd.read_csv('{}.csv'.format(dataset_name))

    column = ['smiles'] + qm9_tasks
    data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
    data_pd.columns = ['SMILES'] + qm9_tasks
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
    data_pd = data_pd[['SMILES', 'Fingerprints'] + qm9_tasks]

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    print('total shape\t', data_pd.shape)

    suppl = Chem.SDMolSupplier('{}.sdf'.format(dataset_name), clean_mols, False, False)
    molecule_list = [mol for mol in suppl]

    raw_df = pd.read_csv('{}.sdf.csv'.format(dataset_name))
    print(raw_df.shape, '\t', raw_df.columns)

    kf = KFold(n_splits=K, shuffle=True)
    for i, (_,index) in enumerate(kf.split(np.arange(data_pd.shape[0]))):
        print(index)

        temp_pd = data_pd.iloc[index]
        print(i, '\t', temp_pd.shape)
        temp_pd.to_csv('{}/{}.csv.gz'.format(dataset_name, i), compression='gzip', index=None)

        w = Chem.SDWriter('{}/{}.sdf'.format(dataset_name, i))
        for id in index:
            w.write(molecule_list[id])
            w.flush()

        temp_pd = raw_df.iloc[index]
        temp_pd.to_csv('{}/{}.sdf.csv'.format(dataset_name, i), index=None)

    return


if __name__ == '__main__':
    dataset_name = 'qm9'
    prepare(dataset_name)
    print()

    for i in range(K):
        extract_graph_multi_tasks_SDF(data_path='{}/{}.csv.gz'.format(dataset_name, i),
                                      sdf_data_path='{}/{}.sdf'.format(dataset_name, i),
                                      out_file_path='{}/{}_graph.npz'.format(dataset_name, i),
                                      task_list=qm9_tasks,
                                      max_atom_num=max_atom_num)
