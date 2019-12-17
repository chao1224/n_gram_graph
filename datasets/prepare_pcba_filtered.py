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
pcba_tasks = [
    'PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457',
    'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469',
    'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688',
    'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242',
    'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546',
    'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676',
    'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294',
    'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349',
    'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947',
    'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339',
    'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842',
    'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317',
    'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590',
    'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233',
    'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171',
    'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288',
    'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635',
    'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104',
    'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979',
    'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553',
    'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709',
    'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881',
    'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899',
    'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915',
    'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995'
]


def get_hit_ratio():
    dataset2hit_ratio = {}
    dataset2number = {}

    whole_data_pd = pd.read_csv('{}.csv.gz'.format(dataset_name))
    data_pd = whole_data_pd.rename(columns={"smiles": "SMILES", "mol_id": "Molecule"})

    for target_name in pcba_tasks:
        y_label = data_pd[target_name].tolist()
        y_label = np.array(y_label)
        non_missing_idx = np.where((np.isnan(y_label) == False))
        y_label = y_label[non_missing_idx]
        hit_ratio = 1.0 * sum(y_label) / len(y_label)
        dataset2hit_ratio[target_name] = hit_ratio
        dataset2number[target_name] = len(y_label)

    target_name_to_hit_ratio = sorted(dataset2hit_ratio.items(), key=lambda (k,v): v)
    for (k,v) in target_name_to_hit_ratio:
        print('{:15s}\t{:.5f}\t\t{}'.format(k, v, dataset2number[k]))
    print()
    print()
    print()

    target_name_to_hit_ratio = sorted(dataset2hit_ratio.items(), key=lambda (k,v): v)
    for (k,v) in target_name_to_hit_ratio:
        if dataset2number[k] > 100000:
            continue
        print('{:15s}\t{:.5f}\t\t{}'.format(k, v, dataset2number[k]))
    return


if __name__ == '__main__':
    dataset_name = 'pcba'
    get_hit_ratio()
