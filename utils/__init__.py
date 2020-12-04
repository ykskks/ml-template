import os
import time
from contextlib import contextmanager
import logging
import random
import math
from pathlib import Path

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import feather
from lightgbm.callback import _format_eval_result
import torch
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import sweetviz as sv
from pandas_profiling import ProfileReport


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'{name}: finished in {end_time - start_time} s')


def get_logger(config_name):
    # prepare the log file
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    sc = logging.StreamHandler()
    logger.addHandler(sc)
    fh = logging.FileHandler(f"logs/{config_name}.log", 'w+')
    logger.addHandler(fh)
    logger.debug(f"logs/{config_name}.log")
    return logger


def track_experiment(model_id, field, value, csv_file='logs/tracking.csv',
                     integer=False, digits=None):
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except FileNotFoundError:
        df = pd.DataFrame()

    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[model_id, field] = value  # Model number is index
    df.to_csv(csv_file)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_datasets(feats, debug=False, n=1000):
    if debug:
        train_feats = [feather.read_dataframe(f'features/{feat}_train.feather').head(n) for feat in feats]
        test_feats = [feather.read_dataframe(f'features/{feat}_test.feather').head(n) for feat in feats]
    else:
        train_feats = [feather.read_dataframe(f'features/{feat}_train.feather') for feat in feats]
        test_feats = [feather.read_dataframe(f'features/{feat}_test.feather') for feat in feats]

    train = pd.concat(train_feats, axis=1)
    test = pd.concat(test_feats, axis=1)
    return train, test


def load_target(target_name, debug=False, n=1000):
    if debug:
        train = feather.read_dataframe('data/input/train.feather').head(n)
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


def plot_distributions(dfs, plot_cols=None, exclude_cols=None, fig_n_cols=5, **kwargs):
    if not isinstance(dfs, list):
        raise Exception("Input must be a list of pd.DataFrames")

    if len(dfs) > 3:
        raise Exception("Cannot handle more than 3 dfs, the plot will be too messy.")

    # drop nan col
    total_dfs = pd.concat(dfs, axis=0)
    drop_cols = total_dfs.columns.values[pd.isnull(total_dfs).sum(axis=0) > 0]
    for df in dfs:
        df.drop(drop_cols, inplace=True, axis=1)

    # cols指定がなければ、numerical colを取得
    if plot_cols is None:
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        plot_cols = df.select_dtypes(include=numeric_types).columns.tolist()
        if exclude_cols is not None:
            plot_cols = [col for col in plot_cols if col not in exclude_cols]

    # figを定義, axesが1-D arrayになるとindexエラーが出るので二次元に統一
    fig, axes = plt.subplots(math.ceil(len(plot_cols)/fig_n_cols), fig_n_cols, figsize=(16, int(0.5 * len(plot_cols))))
    axes = axes.reshape(-1, fig_n_cols)

    # 描画
    colors = ["darkred", "darkblue", "darkgreen"]
    for i, col in enumerate(plot_cols):
        ax = axes[i//fig_n_cols][i%fig_n_cols]
        for j, df in enumerate(dfs):
            ax.hist(df[col], density=True, color=colors[j], alpha=0.5, **kwargs)
            if df[col].nunique() > 1:
                kde = stats.gaussian_kde(df[col])
                xx = np.linspace(df[col].min(), df[col].max(), 1000)
                ax.plot(xx, kde(xx), color=colors[j], alpha=0.7)
        ax.set_title(col)
    fig.tight_layout()
    plt.show()


def eval_func(preds, data):
    # trues = data.get_label()
    # preds_reorderd = np.transpose(preds.reshape(31, -1))
    # score = top2accuracy(preds_reorderd, trues)
    # return "top2accuracy", score, True
    raise RuntimeError("Please define eval func.")


def top2accuracy(preds, trues):
    pred_labels = np.argsort(preds, axis=1)[:, -2:]
    cnt = 0
    for i in range(pred_labels.shape[0]):
        if trues[i] in pred_labels[i, :]:
            cnt += 1
    score = cnt / pred_labels.shape[0]
    return score


def plot_venn2(train, test, include_cols=None, exclude_cols=None):
    train = train.copy(deep=True)
    test = test.copy(deep=True)
    if exclude_cols is not None:
        train.drop(exclude_cols, axis=1, inplace=True)
        test.drop(exclude_cols, axis=1, inplace=True)
    if include_cols is not None:
        train = train[include_cols]
        test = test[include_cols]

    n_vars = len(train.columns.values)
    fig, axes = plt.subplots(nrows=n_vars//5 + 1, ncols=5, figsize=(16, 6))
    i = 0
    for col in train.columns.values:
        venn2([set(train[col].unique()), set(test[col].unique())], set_labels=["train", "test"], ax=axes[i//5][i%5])
        axes[i//5][i%5].set_title(col)

        only_test_element = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
        only_test_element_ratio = test[col].isin(only_test_element).sum() / len(test)
        axes[i//5][i%5].text(1, 1, round(only_test_element_ratio, 3))

        i += 1


def generate_eda(df_path="../data/raw/", save_path="../data/", mode="all"):
    if mode not in ["sv", "pdp", "all"]:
        raise ValueError("Valid arguments are `sv`, `pdp`, `all`")

    train = pd.read_csv(Path(df_path) / "train.csv")
    test = pd.read_csv(Path(df_path) / "test.csv")

    def generate_sv(train, test):
        my_report = sv.compare([train, "train"], [test, "test"])
        my_report.show_html(Path(save_path) / "sv.html")

    def generate_pdb(train, test):
        train_profile = ProfileReport(train, title="train")
        test_profile = ProfileReport(test, title="test")
        train_profile.to_file(Path(save_path) / "pdb_train.html")
        test_profile.to_file(Path(save_path) / "pdb_test.html")

    if mode == "sv":
        generate_sv(train, test)

    elif mode == "pdb":
        generate_pdb(train, test)

    else:
        generate_sv(train, test)
        generate_pdb(train, test)