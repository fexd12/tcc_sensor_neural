import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from feats import Feats
from load_data import DataLoader


class PreProcessamento(DataLoader):
    def __init__(self, feats: Feats, filename: str, sfreq: int) -> None:
        DataLoader.__init__(self, filename, sfreq)
        self._feats = feats

        self.feature_engineer()

    def process_img(self):
        pass

    def feature_engineer(self, ):
        np.random.seed(self._feats.random_seed)

        self._feats.x_train,
        self._feats.x_test,
        self._feats.y_train,
        self._feats.y_test = train_test_split(self.raw["x"], self.raw["y"],
                                              test_size=self._feats.test_split,
                                              random_state=self._feats.random_seed,
                                              )

        print(self._feats.x_train)

        # compute class weights for uneven classes
        y_ints = [y.argmax() for y in self._feats.y_train]
        self._feats.class_weights = class_weight.compute_class_weight('balanced',
                                                                      np.unique(y_ints),  # type: ignore
                                                                      y_ints)

        # Print some outputs

        # print('Input Shape: ' + str(self._feats.input_shape))
        # print('x_train shape:', self._feats.x_train.shape)
        # print(self._feats.x_train.shape[0], 'train samples')
        # print(self._feats.x_test.shape[0], 'test samples')
        # print(self._feats.x_val.shape[0], 'validation samples')
        # print('Class Weights: ' + str(self._feats.class_weights))
