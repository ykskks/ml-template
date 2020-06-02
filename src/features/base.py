import argparse
import inspect
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
import feather

from utils import timer


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    parser.add_argument('--which', '-w', type=str, help='Which feature to overwrite')
    return parser.parse_args()


class FeatureGenerator:
    def __init__(self, namespace, overwrite, which):
        self.namespace = namespace
        self.overwrite = overwrite
        self.which = which

    def get_features(self):
        for k, v in self.namespace.items():
            if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
                yield v()

    def run(self):
        for feature in self.get_features():
            if feature.train_path.exists() and feature.test_path.exists() and not self.overwrite:
                print(feature.name, 'was skipped.')
            elif feature.train_path.exists() and feature.test_path.exists() and self.overwrite and self.which == feature.name:
                feature.run().save()
            elif feature.train_path.exists() and feature.test_path.exists() and self.overwrite and self.which != feature.name:
                print(feature.name, 'was skipped.')
            else:
                feature.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    base_dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.base_dir) / f'{self.name}_train.feather'
        self.test_path = Path(self.base_dir) / f'{self.name}_test.feather'

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = self.suffix + '_' if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        feather.write_dataframe(self.train, str(self.train_path))
        feather.write_dataframe(self.test, str(self.test_path))

    def load(self):
        self.train = feather.read_dataframe(str(self.train_path))
        self.test = feather.read_dataframe(str(self.test_path))
