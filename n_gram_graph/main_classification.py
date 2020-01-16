from __future__ import print_function

import argparse
import numpy as np
import json

import n_gram_graph
from n_gram_graph import model
from n_gram_graph.util import *
from n_gram_graph.dataset_specification import dataset2task_list


def run_n_gram_xgb():
    from model.xgboost_classification import XGBoostClassification

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = [label_name]
    print('label_name_list ', label_name_list)

    test_index = [running_index]
    train_index = filter(lambda x: x not in test_index, np.arange(5))
    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    X_train, y_train = extract_feature_and_label_npy(train_file_list,
                                                     feature_name='embedded_graph_matrix_list',
                                                     label_name_list=label_name_list,
                                                     n_gram_num=n_gram_num)
    X_test, y_test = extract_feature_and_label_npy(test_file_list,
                                                   feature_name='embedded_graph_matrix_list',
                                                   label_name_list=label_name_list,
                                                   n_gram_num=n_gram_num)
    print('done data preparation')

    task = XGBoostClassification(conf=conf)
    print(X_train.shape, '\t', y_train.shape, '\t', X_test.shape, '\t', y_test.shape)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)
    y_pred_on_test = task.predict_with_existing(X_test, weight_file)
    np.savez('output_on_test', y_test=y_test, y_pred=y_pred_on_test)
    return


def run_n_gram_rf():
    from model.random_forest_classification import RandomForestClassification

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = [label_name]
    print('label_name_list ', label_name_list)

    test_index = [running_index]
    train_index = filter(lambda x: x not in test_index, np.arange(5))
    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    X_train, y_train = extract_feature_and_label_npy(train_file_list,
                                                     feature_name='embedded_graph_matrix_list',
                                                     label_name_list=label_name_list,
                                                     n_gram_num=n_gram_num)
    X_test, y_test = extract_feature_and_label_npy(test_file_list,
                                                   feature_name='embedded_graph_matrix_list',
                                                   label_name_list=label_name_list,
                                                   n_gram_num=n_gram_num)
    print('done data preparation')

    task = RandomForestClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)
    y_pred_on_test = task.predict_with_existing(X_test, weight_file)
    np.savez('output_on_test', y_test=y_test, y_pred=y_pred_on_test)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', required=True)
    parser.add_argument('--weight_file', required=True)
    parser.add_argument('--model',  required=True)
    parser.add_argument('--task', default='NR-AhR')
    parser.add_argument('--n_gram_num', type=int, default=6)
    parser.add_argument('--embedding_dimension', type=int, default=100)
    parser.add_argument('--running_index', type=int, default=0)
    given_args = parser.parse_args()

    config_json_file = given_args.config_json_file
    weight_file = given_args.weight_file
    n_gram_num = given_args.n_gram_num
    embedding_dimension = given_args.embedding_dimension
    running_index = given_args.running_index

    K = 5
    task = given_args.task
    model = given_args.model
    label_name = task

    if task in dataset2task_list['tox21']:
        dataset = 'tox21'
        dir_ = '{}/{}'.format(dataset, task)
    elif task in dataset2task_list['clintox']:
        dataset = 'clintox'
        dir_ = '{}/{}'.format(dataset, task)
    elif task in dataset2task_list['muv']:
        dataset = 'muv'
        dir_ = '{}/{}'.format(dataset, task)
    else:
        dataset = task
        dir_ = '{}'.format(dataset)

    if 'n_gram' in model:
        directory = '../datasets/{}/{}/{{}}_grammed_cbow_{}_graph.npz'.format(dir_, running_index, embedding_dimension)
        label_name = 'label_name'
    else:
        directory = '../datasets/{}/{{}}.csv.gz'.format(dir_)
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list = np.array(file_list)

    if model == 'n_gram_xgb':
        run_n_gram_xgb()
    elif model == 'n_gram_rf':
        run_n_gram_rf()
    else:
        raise Exception('No such model! Should be among [{}, {}].'.format(
            'n_gram_xgb',
            'n_gram_rf'
        ))

    import os
    os.rename('output_on_test.npz', '../output/{}/{}/{}.npz'.format(model, running_index, task))
