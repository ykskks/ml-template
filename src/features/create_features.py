import sys
from pathlib import Path

import feather

sys.path.append('.')
from base import get_arguments, FeatureGenerator, Feature


Feature.base_dir = 'src/features'


class Example(Feature):
    pass


if __name__ == '__main__':
    TRAIN_PATH = Path("./data/raw/train.feather")
    TEST_PATH = Path("./data/raw/test.feather")

    args = get_arguments()

    train = feather.read_dataframe(TRAIN_PATH)
    test = feather.read_dataframe(TEST_PATH)

    FeatureGenerator(globals(), args.force, args.which).run()
