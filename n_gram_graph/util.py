from __future__ import print_function

import numpy as np
import pandas as pd


def extract_feature_and_label_npy(data_file_list, feature_name, label_name_list, n_gram_num):
    X_data = []
    y_data = []
    for data_file in data_file_list:
        data = np.load(data_file)
        X_data_temp = data[feature_name]
        if X_data_temp.ndim == 4:
            print('original size\t', X_data_temp.shape)
            X_data_temp = X_data_temp[:, :n_gram_num, ...]
            print('truncated size\t', X_data_temp.shape)

            molecule_num, _, embedding_dimension, segmentation_num = X_data_temp.shape

            X_data_temp = X_data_temp.reshape((molecule_num, n_gram_num*embedding_dimension*segmentation_num), order='F')

        elif X_data_temp.ndim == 3:
            print('original size\t', X_data_temp.shape)
            X_data_temp = X_data_temp[:, :n_gram_num, ...]
            print('truncated size\t', X_data_temp.shape)
            molecule_num, _, embedding_dimension = X_data_temp.shape
            X_data_temp = X_data_temp.reshape((molecule_num, n_gram_num*embedding_dimension), order='F')


        X_data.extend(X_data_temp)

        y_data_temp = map(lambda x: data[x], label_name_list)
        y_data_temp = np.stack(y_data_temp, axis=1)
        y_data.extend(y_data_temp)

    X_data = np.stack(X_data)
    y_data = np.stack(y_data)

    print('X data\t', X_data.shape)
    print('y data\t', y_data.shape)
    return X_data, y_data


def extract_feature_and_label(data_pd,
                              feature_name,
                              label_name_list):
    X_data = data_pd[feature_name].tolist()
    X_data = map(lambda x: list(x), X_data)
    X_data = np.array(X_data)

    y_data = data_pd[label_name_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    return X_data, y_data


def read_merged_data(input_file_list, usecols=None):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        data_pd = pd.read_csv(input_file)
        print(data_pd.columns)
        data_pd = pd.read_csv(input_file, usecols=usecols)
        whole_pd = whole_pd.append(data_pd)
    return whole_pd


def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data

def filter_out_missing_values(data_pd, label_list):
    filtered_pd = data_pd.dropna(axis=0, how='any', inplace=False, subset=label_list)
    return filtered_pd
