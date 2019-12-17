from __future__ import print_function

import numpy as np


dataset2task_list = {
    'tox21':
        [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ],
    'muv':
        [

            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ],
    'hiv': ['hiv'],
    'clintox': ['CT_TOX', 'FDA_APPROVED'],
}

if __name__ == '__main__':
    dataset_list = ['tox21', 'clintox', 'muv', 'hiv']

    for dataset in dataset_list:
        task_list = dataset2task_list[dataset]
        print('{|c|c|c|c|}')
        print('\\hline')
        print('Task & Num of Positives & Total Number & Positive Ratio (\\%)\\\\')
        print('\\hline')
        print('\\hline')

        for task in task_list:
            row = '{}'.format(task)
            if dataset != task:
                data_path = '{}/{}/{{}}_graph.npz'.format(dataset, task)
            else:
                data_path = '{}/{{}}_graph.npz'.format(dataset)

            y_label = []
            for i in range(5):
                data = np.load(data_path.format(i))
                y_label.extend(data['label_name'])
            y_label = np.stack(y_label)

            pos, total = int(sum(y_label)), len(y_label)
            pos_ratio = 100.0 * pos / total
            row = '{}&{}&{}&{:.5f}'.format(row, pos, total, pos_ratio)
            print(row, '\\\\')
        print('\\hline')

        print()
        print()
        print()
