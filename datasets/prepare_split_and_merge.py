from __future__ import print_function

import pandas as pd
import numpy as np
import os

np.random.seed(123)


def split(task, K, N, output_dir='{}_neo_sum'):
    data = np.load('{}/{}_graph.npz'.format(task, K))
    print(data.keys())

    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    bond_attribute_matrix_list = data['bond_attribute_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']
    label_name = data['label_name']
    print(adjacent_matrix_list.shape)
    print(distance_matrix_list.shape)
    print(bond_attribute_matrix_list.shape)
    print(node_attribute_matrix_list.shape)
    print(label_name.shape)

    index_split = np.array_split(range(label_name.shape[0]), N)
    for i, split in enumerate(index_split):
        index = K*N + i
        print(index, '\t', split.shape)

        temp_adjacent_matrix_list = adjacent_matrix_list[split]
        temp_distance_matrix_list = distance_matrix_list[split]
        temp_node_attribute_matrix_list = node_attribute_matrix_list[split]
        temp_label_name = label_name[split]
        print(temp_adjacent_matrix_list.shape)
        print(temp_distance_matrix_list.shape)
        print(temp_node_attribute_matrix_list.shape)
        print(temp_label_name.shape)

        output_path = output_dir + '/{}_graph.npz'.format(index)
        print('task: \t', task, '\t', output_path)
        np.savez_compressed(output_path,
                            adjacent_matrix_list=temp_adjacent_matrix_list,
                            distance_matrix_list=temp_distance_matrix_list,
                            bond_attribute_matrix_list=bond_attribute_matrix_list,
                            node_attribute_matrix_list=temp_node_attribute_matrix_list,
                            label_name=temp_label_name)

def merge(task, K, N, in_dir, out_dir):
    adjacent_matrix_list = []
    distance_matrix_list = []
    node_attribute_matrix_list = []
    random_projected_distance_list = []
    random_projected_adjacent_list = []
    label_name = []

    for j in range(N):
        index = K*N + j
        # print(index)

        data = np.load(in_dir+'/{}_grammed_random_50_graph.npz'.format(index))

        temp_adjacent_matrix_list = data['adjacent_matrix_list']
        temp_distance_matrix_list = data['distance_matrix_list']
        temp_node_attribute_matrix_list = data['node_attribute_matrix_list']
        temp_random_projected_adjacent_list = data['random_projected_adjacent_list']
        temp_random_projected_distance_list = data['random_projected_distance_list']
        temp_label_name = data['label_name']
        print(temp_adjacent_matrix_list.shape, '\t',
              temp_distance_matrix_list.shape, '\t',
              temp_node_attribute_matrix_list.shape, '\t',
              temp_random_projected_adjacent_list.shape, '\t',
              temp_random_projected_distance_list.shape, '\t',
              temp_label_name.shape)

        adjacent_matrix_list.extend(temp_adjacent_matrix_list)
        distance_matrix_list.extend(temp_distance_matrix_list)
        node_attribute_matrix_list.extend(temp_node_attribute_matrix_list)
        random_projected_adjacent_list.extend(temp_random_projected_adjacent_list)
        random_projected_distance_list.extend(temp_random_projected_distance_list)
        label_name.extend(temp_label_name)

    adjacent_matrix_list = np.array(adjacent_matrix_list)
    distance_matrix_list = np.array(distance_matrix_list)
    node_attribute_matrix_list = np.array(node_attribute_matrix_list)
    random_projected_distance_list = np.array(random_projected_distance_list)
    random_projected_adjacent_list = np.array(random_projected_adjacent_list)
    label_name = np.array(label_name)

    print(random_projected_distance_list.shape)
    print(random_projected_adjacent_list.shape)

    output_path = out_dir+ '/{}_grammed_random_50_graph.npz'.format(K)
    print(output_path)
    np.savez_compressed(
        output_path,
        adjacent_matrix_list=adjacent_matrix_list,
        distance_matrix_list=distance_matrix_list,
        node_attribute_matrix_list=node_attribute_matrix_list,
        random_projected_distance_list=random_projected_distance_list,
        random_projected_adjacent_list=random_projected_adjacent_list,
        label_name=label_name
    )


if __name__ == '__main__':
    # for k in range(5):
    #     split(task='malaria', K=k, N=10)
    # for k in range(5):
    #     split(task='cep', K=k, N=30)
    #
    # for k in range(5):
    #     merge(task='malaria', K=k, N=10)
    # for k in range(5):
    #     merge(task='cep', K=k, N=30)

    task_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                  'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    # for task_name in task_names:
    #     print('task name\t', task_name)
    #     for k in range(5):
    #         split(task='tox21/{}'.format(task_name), K=k, N=10,
    #               output_dir='tox21_neo_sum/{}'.format(task_name))

    count = 0
    for tid, task_name in enumerate(task_names):
        print()
        print('task name\t', task_name)
        for k in range(5):
            merge(task=task_name, K=k, N=10,
                  in_dir='tox21_neo_sum/{}'.format(task_name),
                  out_dir='tox21_neo/{}'.format(task_name))
    print('missing in all\t', count)