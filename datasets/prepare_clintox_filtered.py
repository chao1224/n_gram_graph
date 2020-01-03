from __future__ import print_function

import pandas as pd

np.random.seed(123)
clintox_tasks = [u'FDA_APPROVED', u'CT_TOX']


def get_hit_ratio():
    dataset2hit_ratio = {}
    dataset2number = {}

    whole_data_pd = pd.read_csv('{}.csv.gz'.format(dataset_name))
    print(whole_data_pd.columns)
    data_pd = whole_data_pd.rename(columns={"smiles": "SMILES", "mol_id": "Molecule"})

    for target_name in clintox_tasks:
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

    return


if __name__ == '__main__':
    dataset_name = 'clintox'
    get_hit_ratio()
