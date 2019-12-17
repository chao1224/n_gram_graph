from __future__ import print_function

import argparse
import numpy as np
import copy
import collections
from sklearn.svm import SVC, SVR
import time


class Lookup:
    def __init__(self, data=None):
        self.count = 0
        self.table = {}

        # preloaded data
        if data is not None:
            for x in data:
                self[x]

    def __getitem__(self, x):
        if x in self.table:
            return self.table[x]
        else:
            self.table[x] = v = self.count
            self.count += 1
            return v

    @property
    def total(self):
        return self.count

    def clear(self):
        self.count = 0
        self.table = {}


def extract_feature(data_path, label_name='label_name'):
    data = np.load(data_path)
    print(data.keys())
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    bond_attribute_matrix_list = data['bond_attribute_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']
    label_name = data[label_name]

    return adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list,\
           node_attribute_matrix_list, label_name


def get_data(data_path_list):
    adjacent_matrix_, node_attribute_matrix_, label_ = [], [], []

    for data_path in data_path_list:
        adjacent_matrix_list, distance_matrix_list, bond_attribute_matrix_list, node_attribute_matrix_list, label_name = \
            extract_feature(data_path)
        adjacent_matrix_.append(adjacent_matrix_list)
        node_attribute_matrix_.append(node_attribute_matrix_list)
        label_.append(label_name)

    adjacent_matrix_ = np.concatenate(adjacent_matrix_, axis=0)
    node_attribute_matrix_ = np.concatenate(node_attribute_matrix_, axis=0)
    node_attribute_matrix_ = node_attribute_matrix_.astype(str)
    label_ = np.concatenate(label_, axis=0)

    neo_node_attribute_matrix_ = [['' for _ in range(node_attribute_matrix_.shape[1])] for _ in range(node_attribute_matrix_.shape[0])]
    for i,graph in enumerate(node_attribute_matrix_):
        for j,edge in enumerate(node_attribute_matrix_[i]):
            node_attribute = ''.join(node_attribute_matrix_[i][j])
            neo_node_attribute_matrix_[i][j] = node_attribute[:]
    neo_node_attribute_matrix_ = np.array(neo_node_attribute_matrix_)
    print(neo_node_attribute_matrix_.shape)
    return adjacent_matrix_, neo_node_attribute_matrix_, label_


def weisleifer_lehman_graph_kernel(adjacent_matrix_list, node_attribute_matrix_list, h=1):
    start_time = time.time()
    N = adjacent_matrix_list.shape[0]
    max_node_num = adjacent_matrix_list.shape[1]

    lookup = Lookup()
    labels = [np.array([lookup[node_attribute] for node_attribute in node_attribute_matrix]) for node_attribute_matrix in node_attribute_matrix_list]
    labels = np.array(labels)
    print('labels shape\t', labels.shape)

    phi = np.zeros((lookup.total, N))
    for i in range(N):
        for l, counts in collections.Counter(labels[i]).items():
            phi[l, i] = counts
    kernel = np.dot(phi.transpose(), phi)
    print(u'\phi shape: {}\tkernel shape: {}'.format(phi.shape, kernel.shape))

    def long_label(node_attributes, i, neighbour_node_index):
        return tuple((node_attributes[i], tuple(sorted(node_attributes[neighbour_node_index]))))

    for step in range(h):
        print('step: {}'.format(step))
        lookup = Lookup()
        neo_labels = [['' for _ in range(max_node_num)] for _ in range(N)]

        for i in range(N):
            for j in range(max_node_num):
                adjcency = np.nonzero(adjacent_matrix_list[i][j])[0]
                if len(adjcency) == 0:
                    continue
                neo_long_label = long_label(labels[i], j, adjcency)
                neo_labels[i][j] = lookup[neo_long_label]
                # print(neo_long_label, '\t', neo_labels[i][j])

        # feature mapping
        phi = np.zeros((lookup.total, N))
        print(u'\phi shape: {}'.format(phi.shape))
        for i in range(N):
            for l, counts in collections.Counter(neo_labels[i]).items():
                if l == '':
                    continue
                phi[l, i] = counts

        kernel += np.dot(phi.transpose(), phi)
        labels =  np.array(neo_labels)
        print(u'labels shape: {}'.format(labels.shape))
        print()

    # Compute the normalized version of the kernel
    normalized_kernel = np.zeros(kernel.shape)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            normalized_kernel[i, j] = kernel[i, j] / np.sqrt(kernel[i, i] * kernel[j, j])

    end_time = time.time()
    print('Kernel Construction time: {}'.format(end_time - start_time))

    return normalized_kernel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tox21/NR-ER')
    parser.add_argument('--h', type=int, default=1)
    args = parser.parse_args()
    mode = args.mode
    h = args.h

    data_path_list = ['../datasets/{}/{}_graph.npz'.format(mode, i) for i in range(1)]
    adjacent_matrix_list, node_attribute_matrix_list, label_name = get_data(data_path_list)
    print('adjacent_matrix_list:\t\t', adjacent_matrix_list.shape)
    print('node_attribute_matrix_list:\t', node_attribute_matrix_list.shape)
    print('label_name:\t\t\t', label_name.shape)
    print()

    wl_kernel = weisleifer_lehman_graph_kernel(adjacent_matrix_list, node_attribute_matrix_list, h)
    svm_clf = SVC(kernel='precomputed', probability=True)
    svm_clf.fit(wl_kernel, label_name)

    y_pred = svm_clf.predict_proba(wl_kernel)
    y_pred = svm_clf.predict(wl_kernel)
    print(sum(y_pred == label_name))