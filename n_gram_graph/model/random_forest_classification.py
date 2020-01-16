from __future__ import print_function

import argparse
import numpy as np

import n_gram_graph
from n_gram_graph.util import *
from n_gram_graph.evaluation import *
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassification:
    def __init__(self, conf):
        self.conf = conf
        self.max_features = conf['max_features']
        self.n_estimators = conf['n_estimators']
        self.min_samples_leaf = conf['min_samples_leaf']
        self.class_weight = conf['class_weight']
        self.random_seed = conf['random_seed']

        np.random.seed(seed=self.random_seed)
        return
    
    def setup_model(self):
        model = RandomForestClassifier(n_estimators=self.n_estimators,
                                       max_features=self.max_features,
                                       min_samples_leaf=self.min_samples_leaf,
                                       n_jobs=8,
                                       class_weight=self.class_weight,
                                       random_state=self.random_seed,
                                       oob_score=False,
                                       verbose=1)
        return model

    def train_and_predict(self, X_train, y_train, X_test, y_test, weight_file):
        model = self.setup_model()
        model.fit(X_train, y_train)

        y_pred_on_train = reshape_data_into_2_dim(model.predict_proba(X_train)[:, 1])
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict_proba(X_test)[:, 1])
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test)
        self.save_model(model, weight_file)
        return

    def predict_with_existing(self, X_data, weight_file):
        model = self.load_model(weight_file)
        y_pred = reshape_data_into_2_dim(model.predict_proba(X_data)[:, 1])
        return y_pred

    def eval_with_existing(self, X_train, y_train, X_test, y_test, weight_file):
        model = self.load_model(weight_file)

        y_pred_on_train = reshape_data_into_2_dim(model.predict_proba(X_train)[:, 1])
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict_proba(X_test)[:, 1])
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test)
        return

    def save_model(self, model, weight_file):
        from sklearn.externals import joblib
        joblib.dump(model, weight_file, compress=3)
        return

    def load_model(self, weight_file):
        from sklearn.externals import joblib
        model = joblib.load(weight_file)
        return model


def demo_random_forest_classification():
    conf = {
        'max_features': 'log2',
        'n_estimators': 4000,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'random_seed': 1337,
        'label_name_list': ['Keck_Pria_AS_Retest']
    }

    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    test_index = 0
    complete_index = np.arange(K)
    train_index = np.where(complete_index != test_index)[0]
    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index:test_index + 1]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    train_pd = read_merged_data(train_file_list)
    test_pd = read_merged_data(test_file_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)

    task = RandomForestClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file

    # specify dataset
    K = 5
    directory = '../datasets/keck_pria_lc/{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list = np.array(file_list)

    demo_random_forest_classification()
