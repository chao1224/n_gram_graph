from __future__ import print_function

import argparse
import numpy as np

import n_gram_graph
from n_gram_graph.util import *
from n_gram_graph.evaluation import *
from xgboost import XGBClassifier


def eval_precision_auc_single(y_predicted, y_true):
    y_true = y_true.get_label()
    value = precision_auc_single(y_predicted, y_true)
    return 'AUC[PR]', 1-value


def eval_roc_auc_single(y_predicted, y_true):
    y_true = y_true.get_label()
    value = roc_auc_single(y_predicted, y_true)
    return 'AUC[ROC]', 1-value


eval_metric_mapping = {'AUC[ROC]': eval_roc_auc_single, 'AUC[PR]': eval_precision_auc_single}


class XGBoostClassification:
    def __init__(self, conf):
        self.conf = conf
        self.max_depth = conf['max_depth']
        self.learning_rate = conf['learning_rate']
        self.n_estimators = conf['n_estimators']
        self.objective = conf['objective']
        self.booster = conf['booster']
        self.subsample = conf['subsample']
        self.colsample_bylevel = conf['colsample_bylevel']
        self.colsample_bytree = conf['colsample_bytree']
        self.min_child_weight = conf['min_child_weight']
        self.reg_alpha = conf['reg_alpha']
        self.reg_lambda = conf['reg_lambda']
        self.scale_pos_weight = conf['scale_pos_weight']
        self.max_delta_step = conf['max_delta_step']

        self.random_seed = conf['random_seed']
        self.eval_metric = eval_metric_mapping[conf['early_stopping']['eval_metric']]
        self.early_stopping_round = conf['early_stopping']['round']

        np.random.seed(seed=self.random_seed)
        return
    
    def setup_model(self):
        model = XGBClassifier(max_depth=self.max_depth,
                              learning_rate=self.learning_rate,
                              n_estimators=self.n_estimators,
                              objective=self.objective,
                              booster=self.booster,
                              subsample=self.subsample,
                              colsample_bylevel=self.colsample_bylevel,
                              colsample_bytree=self.colsample_bytree,
                              min_child_weight=self.min_child_weight,
                              reg_alpha=self.reg_alpha,
                              reg_lambda=self.reg_lambda,
                              scale_pos_weight=self.scale_pos_weight,
                              max_delta_step=self.max_delta_step,
                              random_state=self.random_seed,
                              silent=False,
                              n_jobs=8)
        return model

    def train_and_predict(self, X_train, y_train, X_test, y_test, weight_file):
        model = self.setup_model()
        model.fit(X_train, y_train, verbose=True)

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


def demo_xgboost_classification():
    conf = {
        'max_depth': 10,
        'learning_rate': 1e-1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'subsample': 1,
        'colsample_bylevel': 1,
        'colsample_bytree': 1,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'max_delta_step': 1,

        'early_stopping': {
            'eval_metric': 'AUC[PR]',
            'round': 300,
        },
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        "random_seed": 1337,
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

    task = XGBoostClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
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

    demo_xgboost_classification()
