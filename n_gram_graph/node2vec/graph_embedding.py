from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from node_embedding import CBoW


mode2task_list = {
    'qm8': [
        'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
        'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
    ],
    'qm9': [
        'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
        'u298_atom', 'h298_atom', 'g298_atom'
    ]
}


def get_data(data_path):
    data = np.load(data_path)
    print(data.keys())
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    bond_attribute_matrix_list = data['bond_attribute_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    kwargs = {}
    if mode in ['qm8', 'qm9']:
        for task in mode2task_list[mode]:
            kwargs[task] = data[task]
    else:
        kwargs['label_name'] = data['label_name']

    return adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list,\
           node_attribute_matrix_list, kwargs


class GraphDataset(Dataset):
    def __init__(self, node_attribute_matrix_list, adjacent_matrix_list, distance_matrix_list):
        self.node_attribute_matrix_list = node_attribute_matrix_list
        self.adjacent_matrix_list = adjacent_matrix_list
        self.distance_matrix_list = distance_matrix_list

    def __len__(self):
        return len(self.node_attribute_matrix_list)

    def __getitem__(self, idx):
        node_attribute_matrix = torch.from_numpy(self.node_attribute_matrix_list[idx])
        adjacent_matrix = torch.from_numpy(self.adjacent_matrix_list[idx])
        distance_matrix = torch.from_numpy(self.distance_matrix_list[idx])
        return node_attribute_matrix, adjacent_matrix, distance_matrix


def get_walk_representation(dataloader):
    X_embed = []
    random_projected_list = []
    for batch_id, (node_attribute_matrix, adjacent_matrix, distance_matrix) in enumerate(dataloader):
        node_attribute_matrix = Variable(node_attribute_matrix).float()
        adjacent_matrix = Variable(adjacent_matrix).float()
        distance_matrix = Variable(distance_matrix).float()
        if torch.cuda.is_available():
            node_attribute_matrix = node_attribute_matrix.cuda()
            adjacent_matrix = adjacent_matrix.cuda()
            distance_matrix = distance_matrix.cuda()

        tilde_node_attribute_matrix = model.embeddings(node_attribute_matrix)

        walk = tilde_node_attribute_matrix
        v1 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v2 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v3 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v4 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v5 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v6 = torch.sum(walk, dim=1)

        random_projected_matrix = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)

        if torch.cuda.is_available():
            tilde_node_attribute_matrix = tilde_node_attribute_matrix.cpu()
            random_projected_matrix = random_projected_matrix.cpu()
        X_embed.extend(tilde_node_attribute_matrix.data.numpy())
        random_projected_list.extend(random_projected_matrix.data.numpy())

    embedded_node_attribute_matrix_list = np.array(X_embed)
    random_projected_list = np.array(random_projected_list)
    print('embedded_node_attribute_matrix_list: ', embedded_node_attribute_matrix_list.shape)
    print('random_projected_list shape: {}'.format(random_projected_list.shape))

    return embedded_node_attribute_matrix_list, random_projected_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='delaney')
    parser.add_argument('--running_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    mode = args.mode
    running_index = args.running_index
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    random_dimension_list = [50, 100]
    if mode in ['hiv'] or 'pcba' in mode or 'clintox' in mode:
        feature_num = 42
        max_atom_num = 100
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                             range(36, 38), range(38, 40), range(40, 42)]
    elif mode in ['qm8', 'qm9']:
        feature_num = 32
        max_atom_num = 55
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 32)]
    else:
        feature_num = 42
        max_atom_num = 55
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                             range(36, 38), range(38, 40), range(40, 42)]

    segmentation_list = np.array(segmentation_list)
    segmentation_num = len(segmentation_list)

    test_list = [running_index]
    train_list = filter(lambda x: x not in test_list, np.arange(5))
    print('training list: {}\ttest list: {}'.format(train_list, test_list))

    for random_dimension in random_dimension_list:
        model = CBoW(feature_num=feature_num, embedding_dim=random_dimension,
                     task_num=segmentation_num, task_size_list=segmentation_list)

        weight_file = 'model_weight/{}/{}/{}_CBoW_non_segment.pt'.format(mode, running_index, random_dimension)
        print('weight file is {}'.format(weight_file))
        model.load_state_dict(torch.load(weight_file))
        if torch.cuda.is_available():
            model.cuda()
        # print(model)
        model.eval()

        start_time = time.time()
        for i in range(5):
            data_preprocess_start_time = time.time()
            data_path = '../../datasets/{}/{}_graph.npz'.format(mode, i)
            adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list, node_attribute_matrix_list, kwargs = get_data(data_path)
            dataset = GraphDataset(node_attribute_matrix_list=node_attribute_matrix_list, adjacent_matrix_list=adjacent_matrix_list, distance_matrix_list=distance_matrix_list)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            data_preprocess_end_time = time.time()
            print('Data preprocessing time: {}'.format(data_preprocess_end_time - data_preprocess_start_time))

            embedded_node_attribute_matrix_list, random_projected_list = get_walk_representation(dataloader)
            print('random_projected_list\t', random_projected_list.shape)

            out_file_path = '../../datasets/{}/{}/{}_grammed_cbow_{}_graph'.format(mode, running_index, i, random_dimension)
            kwargs['adjacent_matrix_list'] = adjacent_matrix_list
            kwargs['distance_matrix_list'] = distance_matrix_list
            kwargs['node_attribute_matrix_list'] = embedded_node_attribute_matrix_list
            kwargs['random_projected_list'] = random_projected_list
            np.savez_compressed(out_file_path, **kwargs)
            print(kwargs.keys())
            print()
        end_time = time.time()
        processing_time = end_time - start_time
        print('For random dimension as {}, the processing time is {}.'.format(random_dimension, processing_time))
        print()
        print()
        print()
