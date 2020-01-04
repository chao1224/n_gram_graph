from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def roc_auc_single(predicted, actual):
    try:
        auc_ret = roc_auc_score(actual, predicted)
    except ValueError:
        auc_ret = np.nan

    return auc_ret


def precision_auc_single(predicted, actual):
    try:
        prec_auc = average_precision_score(actual, predicted)
    except ValueError:
        prec_auc = np.nan
    return prec_auc


def rms_score(y_true, y_pred):
  """Computes RMS error."""
  return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true, y_pred):
  """Computes MAE."""
  return mean_absolute_error(y_true, y_pred)


def pearson_r2_score(y, y_pred):
    """Computes Pearson R^2 (square of Pearson correlation)."""
    y, y_pred = np.squeeze(y), np.squeeze(y_pred)
    return pearsonr(y, y_pred)[0] ** 2


def output_classification_result(y_train, y_pred_on_train,
                                 y_val, y_pred_on_val,
                                 y_test, y_pred_on_test):

    if y_pred_on_train is not None:
        print('train precision: {}'.format(precision_auc_single(y_pred_on_train, y_train)))
        print('train roc: {}'.format(roc_auc_single(y_pred_on_train, y_train)))
        print()

    if y_pred_on_val is not None:
        print('val precision: {}'.format(precision_auc_single(y_pred_on_val, y_val)))
        print('val roc: {}'.format(roc_auc_single(y_pred_on_val, y_val)))
        print()

    if y_pred_on_test is not None:
        print('test precision: {}'.format(precision_auc_single(y_pred_on_test, y_test)))
        print('test roc: {}'.format(roc_auc_single(y_pred_on_test, y_test)))
        print()

    return


def output_regression_result(y_train, y_pred_on_train,
                             y_val, y_pred_on_val,
                             y_test, y_pred_on_test):
    def output(y_true, y_pred, mode):
        pearson_r2 = pearson_r2_score(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rms = rms_score(y_true, y_pred)
        mae = mae_score(y_true, y_pred)
        print('Pearson R2 on {}: {}'.format(mode, pearson_r2))
        print('R2 on {}: {}'.format(mode, r2))
        print('RMSE on {}: {}'.format(mode, rms))
        print('MAE on {}: {}'.format(mode, mae))
        print()
        return

    if y_pred_on_train is not None:
        output(y_train, y_pred_on_train, 'train set')
    if y_pred_on_val is not None:
        output(y_val, y_pred_on_val, 'val set')
    if y_pred_on_test is not None:
        output(y_test, y_pred_on_test, 'test set')
    return
