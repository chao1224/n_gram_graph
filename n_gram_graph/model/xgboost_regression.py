from __future__ import print_function

import argparse
import numpy as np

import n_gram_graph
from n_gram_graph.util import *
from n_gram_graph.evaluation import *
from xgboost import XGBRegressor


class XGBoostRegression:
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

        self.random_seed = conf['random_seed']

        np.random.seed(seed=self.random_seed)
        return

    def setup_model(self):
        model = XGBRegressor(max_depth=self.max_depth,
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
                             random_state=self.random_seed,
                             silent=False,
                             n_jobs=8)
        return model

    def train_and_predict(self, X_train, y_train, X_test, y_test, weight_file):
        model = self.setup_model()
        model.fit(X_train, y_train, verbose=True)

        y_pred_on_train = reshape_data_into_2_dim(model.predict(X_train))
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict(X_test))

        output_regression_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                 y_val=None, y_pred_on_val=None,
                                 y_test=y_test, y_pred_on_test=y_pred_on_test)
        self.save_model(model, weight_file)
        return

    def predict_with_existing(self, X_data, weight_file):
        model = self.load_model(weight_file)
        y_pred = reshape_data_into_2_dim(model.predict(X_data))
        return y_pred

    def eval_with_existing(self, X_train, y_train, X_test, y_test, weight_file):
        model = self.load_model(weight_file)

        y_pred_on_train = reshape_data_into_2_dim(model.predict(X_train))
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict(X_test))
            
        output_regression_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
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


def demo_xgboost_regression():
    conf = {
        'max_depth': 10,
        'learning_rate': 1e-1,
        'n_estimators': 100,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'subsample': 1,
        'colsample_bylevel': 1,
        'colsample_bytree': 1,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'random_seed': 1337,
        'label_name_list': ['delaney']
    }

    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    test_index = 0
    train_index = slice(1, 5)

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
    print('done data preparation')

    task = XGBoostRegression(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file

    # specify dataset
    K = 5
    directory = '../datasets/delaney/{}.csv.gz'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))

    demo_xgboost_regression()
