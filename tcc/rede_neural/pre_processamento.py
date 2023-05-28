# import os
import numpy as np
from tcc.utils.feats import Feats
from tcc.rede_neural.load_data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


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

        # if not self.cache_filename:
        #     print("not set cache, get from raw dataset...")
        #     x, y = self._read_files()
        # else:
        #     if len(os.listdir(self.cache_filename)) > 0:
        #         print("reading cache...")
        #         try:
        #             data_cached = np.load(self.cache_filename + "cache.npz", 'r', allow_pickle=True)
        #             print(data_cached.files)
        #             x = data_cached['x'] or []
        #             y = data_cached['y'] or []
        #         except Exception as e:
        #             print("Failed to load cache file", e)
        #     else:
        #         print("cache not found, reading raw dataset...")
        #         # print(self._read_files())
        #         x, y = self._read_files()

        #         np.savez(self.cache_filename + 'cache.npz',
        #                  x=x,
        #                  y=y
        #                  )

        # print(len(*data))

        # print(len(train))
        # print(type(train))
        # print(train[0][0].shape, train[0][1].shape)

        # list(zip(*train)))
        # map(list, zip(*test))

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

        # self._feats.x_train, \
        #     self._feats.x_test, \
        #     self._feats.y_train, \
        #     self._feats.y_test = train_test_split(x_train, y_train,
        #                                           test_size=self._feats.test_split,
        #                                           random_state=self._feats.random_seed,
        #                                           )
        # # compute class weights for uneven classes
        # y_ints = [y.argmax() for y in np.asarray(self._feats.y_train)]

        # self._feats.class_weights = compute_class_weight(class_weight='balanced',
        #                                                  classes=np.unique(y_ints),
        #                                                  y=y_ints)
