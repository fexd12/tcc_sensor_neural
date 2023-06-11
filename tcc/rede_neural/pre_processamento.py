# import os
import numpy as np
from tcc.utils.feats import Feats
from tcc.rede_neural.load_data import DataLoader


class PreProcessamento(DataLoader):
    def __init__(self, feats: Feats, filename: str, cache_filename=None) -> None:
        DataLoader.__init__(self, filename, feats)

        self.cache_filename = cache_filename
        # self.train_data = []
        # self.test_data = []

        self.feature_engineer()

    def feature_engineer(self):
        print('spliting features...')

        train, test = self._read_files()

        x_train, y_train = map(list, zip(*train))
        x_test, y_test = map(list, zip(*test))

        print(len(x_train), len(y_train))
        print(type(x_train), type(y_train))
        print(x_train[0].shape, y_train[0].shape)
        print(y_train[0])

        self._feats.x_train = np.array(x_train)
        self._feats.x_test = np.array(x_test)
        self._feats.y_train = y_train
        self._feats.y_test = y_test
