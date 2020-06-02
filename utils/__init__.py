import time
from contextlib import contextmanager
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import feather
from lightgbm.callback import _format_eval_result


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'{name}: finished in {end_time - start_time} s')


def load_datasets(feats, debug=False):
    if debug:
        train_feats = [feather.read_dataframe(f'features/{feat}_train.feather').head(10000) for feat in feats]
        train = pd.concat(train_feats, axis=1)
        test_feats = [feather.read_dataframe(f'features/{feat}_test.feather').head(10000) for feat in feats]
        test = pd.concat(test_feats, axis=1)
        return train, test
    else:
        train_feats = [feather.read_dataframe(f'features/{feat}_train.feather') for feat in feats]
        train = pd.concat(train_feats, axis=1)
        test_feats = [feather.read_dataframe(f'features/{feat}_test.feather') for feat in feats]
        test = pd.concat(test_feats, axis=1)
        return train, test


def load_target(target_name, debug=False):
    if debug:
        train = feather.read_dataframe('data/input/train.feather').head(10000)
        target = train[target_name]
        return target
    else:
        train = feather.read_dataframe('data/input/train.feather')
        target = train[target_name]
        return target


def get_categorical_feats(feats):
    categorical_feats = []
    train, test = load_datasets(feats)

    for col in train.columns:
        categorical_feats.append(col)

    return categorical_feats


def calculate_metric(predicted_values, true_values, coupling_types, floor=1e-9):
    """Inputs should be in numpy.array format"""
    metric = 0
    for coupling_type in np.unique(coupling_types):
        group_mae = mean_absolute_error(true_values[coupling_types == coupling_type], predicted_values[coupling_types == coupling_type])
        metric += np.log(max(group_mae, floor))
    return metric / len(np.unique(coupling_types))


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtype

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
