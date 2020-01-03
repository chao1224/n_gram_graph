from __future__ import print_function

import numpy as np
import math
import os

np.random.seed(123)


def n_gram_graph_with_random_projection(data_path, out_file_path, random_dimension, G):
    data = np.load(data_path)
    print(data.keys())
    print(data_path)
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']
    print(adjacent_matrix_list.shape)
    print(distance_matrix_list.shape)
    print(node_attribute_matrix_list.shape)

    segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                         range(36, 38), range(38, 40), range(40, 42)]
    segmentation_num = len(segmentation_list)

    molecule_num = adjacent_matrix_list.shape[0]
    max_atom_num = adjacent_matrix_list.shape[1]

    random_projected_node_attribute_matrix = []
    for segmentation in segmentation_list:
        g_hat_j = G[..., segmentation]
        node_attribute_j = node_attribute_matrix_list[..., segmentation]
        random_projected_on_segmentation = np.matmul(node_attribute_j, g_hat_j.T)
        print('{}\t{}\t{}'.format(g_hat_j.shape, node_attribute_j.shape, random_projected_on_segmentation.shape))
        random_projected_node_attribute_matrix.append(random_projected_on_segmentation)

    random_projected_node_attribute_matrix = np.stack(random_projected_node_attribute_matrix, axis=-1)
    print('Node attribute matrix shape', node_attribute_matrix_list.shape)
    print('Random projection shape\t', G.shape)
    print('Projected matrix shape\t', random_projected_node_attribute_matrix.shape)

    random_projected_list = []
    for index in range(molecule_num):
        adjacent_matrix = adjacent_matrix_list[index]
        distance_matrix = distance_matrix_list[index]
        tilde_node_attribute_matrix = random_projected_node_attribute_matrix[index]

        # 1-gram
        v1 = np.zeros((random_dimension, segmentation_num))
        neighbor_sum = adjacent_matrix.sum(axis=0)
        for i in range(max_atom_num):
            if neighbor_sum[i] == 0:
                continue
            v1 += tilde_node_attribute_matrix[i]

        # 2-gram
        v2 = np.zeros((random_dimension, segmentation_num))
        for i in range(max_atom_num):
            if neighbor_sum[i] == 0:
                continue
            for j in range(i+1, max_atom_num):
                if adjacent_matrix[i][j] == 0:
                    continue
                v2 += tilde_node_attribute_matrix[i] * tilde_node_attribute_matrix[j]

        # 3-gram
        v3 = np.zeros((random_dimension, segmentation_num))
        for i in range(max_atom_num):
            if neighbor_sum[i] == 0:
                continue
            for j in range(i+1, max_atom_num):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, max_atom_num):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    v3 += tilde_node_attribute_matrix[i] * tilde_node_attribute_matrix[j] * tilde_node_attribute_matrix[k]

        # 4-gram
        v4 = np.zeros((random_dimension, segmentation_num))
        for i in range(max_atom_num):
            if neighbor_sum[i] == 0:
                continue
            for j in range(i+1, max_atom_num):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, max_atom_num):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    for l in range(k+1, max_atom_num):
                        if adjacent_matrix[k][l] == 0:
                            continue
                        v4 += tilde_node_attribute_matrix[i] * tilde_node_attribute_matrix[j] * \
                              tilde_node_attribute_matrix[k] * tilde_node_attribute_matrix[l]

        # 5-gram
        v5 = np.zeros((random_dimension, segmentation_num))
        for i in range(max_atom_num):
            if neighbor_sum[i] == 0:
                continue
            for j in range(i+1, max_atom_num):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, max_atom_num):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    for l in range(k+1, max_atom_num):
                        if adjacent_matrix[k][l] == 0:
                            continue
                        for m in range(l+1, max_atom_num):
                            if adjacent_matrix[l][m] == 0:
                                continue
                            v5 += tilde_node_attribute_matrix[i] * tilde_node_attribute_matrix[j] * \
                                  tilde_node_attribute_matrix[k] * tilde_node_attribute_matrix[l] * \
                                  tilde_node_attribute_matrix[m]

        # 6-gram
        v6 = np.zeros((random_dimension, segmentation_num))
        for i in range(max_atom_num):
            if neighbor_sum[i] == 0:
                continue
            for j in range(i+1, max_atom_num):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, max_atom_num):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    for l in range(k+1, max_atom_num):
                        if adjacent_matrix[k][l] == 0:
                            continue
                        for m in range(l+1, max_atom_num):
                            if adjacent_matrix[l][m] == 0:
                                continue
                            for n in range(m+1, max_atom_num):
                                if adjacent_matrix[m][n] == 0:
                                    continue
                                v6 += tilde_node_attribute_matrix[i] * tilde_node_attribute_matrix[j] * \
                                      tilde_node_attribute_matrix[k] * tilde_node_attribute_matrix[l] * \
                                      tilde_node_attribute_matrix[m] * tilde_node_attribute_matrix[n]
        v = np.stack((v1, v2, v3, v4, v5, v6), axis=0)
        random_projected_list.append(v)

    random_projected_list = np.array(random_projected_list)
    print('Randomized shape\t', random_projected_list.shape)
    if 'label_name' in data.keys():
        np.savez_compressed(out_file_path,
                            adjacent_matrix_list=adjacent_matrix_list,
                            distance_matrix_list=distance_matrix_list,
                            node_attribute_matrix_list=node_attribute_matrix_list,
                            random_projected_list=random_projected_list,
                            label_name=data['label_name'])
    else:
        np.savez_compressed(out_file_path,
                            adjacent_matrix_list=adjacent_matrix_list,
                            distance_matrix_list=distance_matrix_list,
                            node_attribute_matrix_list=node_attribute_matrix_list,
                            random_projected_list=random_projected_list)
    print()
    return


def flatten(data_path, out_file_path):
    data = np.load(data_path)
    print(data.keys())
    random_projected_list = data['random_projected_list']
    print('Randomized shape\t', random_projected_list.shape)

    random_projected_list = map(lambda x: x.flatten('F'), random_projected_list)
    random_projected_list = np.array(random_projected_list)
    print('Randomized shape\t', random_projected_list.shape)
    print()

    if 'label_name' in data.keys():
        np.savez_compressed(out_file_path,
                            random_projected_list=random_projected_list,
                            label_name=data['label_name'])
    else:
        np.savez_compressed(out_file_path,
                            random_projected_list=random_projected_list)
    return


def prepare_n_gram_matrix(data_path, out_file_path):
    data = np.load(data_path)
    print(data.keys())
    print(data_path)
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']
    print(adjacent_matrix_list.shape)
    print(distance_matrix_list.shape)
    print(node_attribute_matrix_list.shape)

    L = adjacent_matrix_list.shape[0]
    n = adjacent_matrix_list.shape[1]

    incidence_matrix_list = []
    for index in range(L):
        adjacent_matrix = adjacent_matrix_list[index]
        distance_matrix = distance_matrix_list[index]

        # 1-gram
        v1 = np.zeros(n)
        neighbor_sum = adjacent_matrix.sum(axis=0)
        for i in range(n):
            if neighbor_sum[i] == 0:
                continue
            temp = map(lambda x: 1 if x in [i] else 0, range(n))
            v1 += np.array(temp)

        # 2-gram
        v2 = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                if adjacent_matrix[i][j] == 0:
                    continue
                temp = map(lambda x: 1 if x in [i, j] else 0, range(n))
                v2 += np.array(temp)

        # 3-gram
        v3 = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, n):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    temp = map(lambda x: 1 if x in [i, j, k] else 0, range(n))
                    v3 += np.array(temp)

        # 4-gram
        v4 = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, n):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    for l in range(k+1, n):
                        if adjacent_matrix[k][l] == 0:
                            continue
                        temp = map(lambda x: 1 if x in [i, j, k, l] else 0, range(n))
                        v4 += np.array(temp)

        # 5-gram
        v5 = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, n):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    for l in range(k+1, n):
                        if adjacent_matrix[k][l] == 0:
                            continue
                        for m in range(l+1, n):
                            if adjacent_matrix[l][m] == 0:
                                continue
                            temp = map(lambda x: 1 if x in [i, j, k, l, m] else 0, range(n))
                            v5 += np.array(temp)

        # 6-gram
        v6 = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                if adjacent_matrix[i][j] == 0:
                    continue
                for k in range(j+1, n):
                    if adjacent_matrix[j][k] == 0:
                        continue
                    for l in range(k+1, n):
                        if adjacent_matrix[k][l] == 0:
                            continue
                        for m in range(l+1, n):
                            if adjacent_matrix[l][m] == 0:
                                continue
                            for o in range(m+1, n):
                                if adjacent_matrix[m][o] == 0:
                                    continue
                                temp = map(lambda x: 1 if x in [i, j, k, l, m, o] else 0, range(n))
                                v6 += np.array(temp)

        v = np.stack((v1, v2, v3, v4, v5, v6), axis=1)
        incidence_matrix_list.append(v)

    incidence_matrix_list = np.array(incidence_matrix_list)
    print('Incidence matrix list shape\t', incidence_matrix_list.shape)

    if 'label_name' in data.keys():
        np.savez_compressed(out_file_path,
                            adjacent_matrix_list=adjacent_matrix_list,
                            distance_matrix_list=distance_matrix_list,
                            node_attribute_matrix_list=node_attribute_matrix_list,
                            incidence_matrix_list=incidence_matrix_list,
                            label_name=data['label_name'])
    else:
        np.savez_compressed(out_file_path,
                            adjacent_matrix_list=adjacent_matrix_list,
                            distance_matrix_list=distance_matrix_list,
                            node_attribute_matrix_list=node_attribute_matrix_list,
                            incidence_matrix_list=incidence_matrix_list)
    return


def save_random_projection(r, d):
    if os.path.exists('random_projection_r_{}.npz'.format(r)):
        return
    G = np.random.randn(r, d) / math.sqrt(d)
    np.savez_compressed('random_projection_r_{}'.format(r), G=G)
    return


def load_random_projection(r):
    data = np.load('random_projection_r_{}.npz'.format(r))
    G = data['G']
    return G


if __name__ == '__main__':
    save_random_projection(r=100, d=42)
    save_random_projection(r=50, d=42)

    ################## Create n-gram with random projection ##################

    for random_dimension in [100]:
        G = load_random_projection(r=random_dimension)

        # for i in range(5):
        #     n_gram_graph_with_random_projection(data_path='./keck_pria_lc/{}_graph.npz'.format(i),
        #                                         out_file_path='./keck_pria_lc/{}_grammed_random_{}_graph.npz'.format(i, random_dimension),
        #                                         random_dimension=random_dimension, G=G)
        #
        # n_gram_graph_with_random_projection(data_path='keck_pria_lc/keck_lc4_graph.npz',
        #                                     out_file_path='keck_pria_lc/keck_lc4_grammed_random_{}_graph.npz'.format(random_dimension),
        #                                     random_dimension=random_dimension, G=G)

        target_names = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
        for target_name in target_names:
            directory = 'muv/{}'.format(target_name)
            for i in range(5):
                n_gram_graph_with_random_projection(data_path='{}/{}_graph.npz'.format(directory, i),
                                                    out_file_path='{}/{}_grammed_random_{}_graph.npz'.format(directory, i, random_dimension),
                                                    random_dimension=random_dimension, G=G)

    ################## Flatten random projection ##################
    for random_dimension in [100]:

        # for i in range(5):
        #     flatten(data_path='./keck_pria_lc/{}_grammed_random_{}_graph.npz'.format(i, random_dimension),
        #             out_file_path='./keck_pria_lc/{}_grammed_flatten_random_{}_graph.npz'.format(i, random_dimension))
        # flatten(data_path='keck_pria_lc/keck_lc4_grammed_random_{}_graph.npz'.format(random_dimension),
        #         out_file_path='keck_pria_lc/keck_lc4_grammed_flatten_random_{}_graph.npz'.format(random_dimension))

        target_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                        'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        for target_name in target_names:
            directory = 'tox21/{}'.format(target_name)
            for i in range(5):
                flatten(data_path='{}/{}_grammed_random_{}_graph.npz'.format(directory, i, random_dimension),
                        out_file_path='{}/{}_grammed_flatten_random_{}_graph.npz'.format(directory, i, random_dimension))

        target_names = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
        for target_name in target_names:
            directory = 'muv/{}'.format(target_name)
            for i in range(5):
                flatten(data_path='{}/{}_grammed_random_{}_graph.npz'.format(directory, i, random_dimension),
                        out_file_path='{}/{}_grammed_flatten_random_{}_graph.npz'.format(directory, i, random_dimension))

    ################## Create n-gram with embedding ##################

    # for i in range(5):
    #     prepare_n_gram_matrix(data_path='./keck_pria_lc/{}_graph.npz'.format(i),
    #                           out_file_path='./keck_pria_lc/{}_grammed_matrix.npz'.format(i))
    # prepare_n_gram_matrix(data_path='keck_pria_lc/keck_lc4_graph.npz',
    #                       out_file_path='keck_pria_lc/keck_lc4_grammed_matrix.npz')
    #
    # target_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    #                 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    # for target_name in target_names:
    #     directory = 'tox21/{}'.format(target_name)
    #     for i in range(6):
    #         prepare_n_gram_matrix(data_path='{}/{}_graph.npz'.format(directory, i),
    #                               out_file_path='{}/{}_grammed_matrix.npz'.format(directory, i))
    #
    # target_names = [
    #     'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
    #     'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
    #     'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
    # ]
    # for target_name in target_names:
    #     directory = 'muv/{}'.format(target_name)
    #     for i in range(6):
    #         prepare_n_gram_matrix(data_path='{}/{}_graph.npz'.format(directory, i),
    #                               out_file_path='{}/{}_grammed_matrix.npz'.format(directory, i))
