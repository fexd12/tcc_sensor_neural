from sklearn.model_selection import train_test_split

from tcc.rede_neural.load_data import DataLoader
from tcc.utils.feats import Feats


class PreProcessamento(DataLoader):
    def __init__(self, feats: Feats, filename: str, sfreq: int) -> None:
        DataLoader.__init__(self, filename, sfreq, feats)
        self._feats = feats

        self.feature_engineer()

    def feature_engineer(self):
        print('skip spliting features...')
        # val_prop = self._feats.val_split / (1 - self._feats.test_split)

        # self._feats.x_train, \
        #     self._feats.x_test, \
        #     self._feats.y_train, \
        #     self._feats.y_test = train_test_split(self.raw,
        #                                           test_size=self._feats.test_split,
        #                                           random_state=self._feats.random_seed,
        #                                           )

        # self._feats.x_train, \
        #     self._feats.x_val, \
        #     self._feats.y_train, \
        #     self._feats.y_val = train_test_split(self._feats.x_train, self._feats.y_train,
        #                                          test_size=val_prop,
        #                                          random_state=self._feats.random_seed
        #                                          )

        # # compute class weights for uneven classes
        # y_ints = [y.argmax() for y in np.asarray(self._feats.y_train)]

        # self._feats.class_weights = class_weight.compute_class_weight(class_weight='balanced',
        #                                                               classes=np.unique(y_ints),  # type: ignore
        #                                                               y=y_ints
        #                                                               )
