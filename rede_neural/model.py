import keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from numpy.random import seed
from rede_neural.feats import Feats
from rede_neural.pre_processamento import PreProcessamento

seed(1017)
set_random_seed(1017)

pd.options.display.precision = 4
pd.options.display.max_columns = None
plt.rcParams["figure.figsize"] = (12, 12)


class ModelEEG(PreProcessamento):
    def __init__(self, feats: Feats, filename: str, sfreq: int) -> None:
        PreProcessamento.__init__(self, feats, filename, sfreq)
        self.model = Sequential()
        self.__create_model()

    def __create_model(self, units=[16, 8, 4, 8, 16], dropout=.25, batch_norm=True,):

        print('Creating ' + self._feats.model_type + ' Model')
        print('Input shape: ' + str(self._feats.input_shape))

        # ---DenseFeedforward Network
        # Makes a hidden layer for each item in units
        if self._feats.model_type == 'NN':

            self.model.add(Flatten(input_shape=self._feats.input_shape))

            for unit in units:
                self.model.add(Dense(unit))
                if batch_norm:
                    self.model.add(BatchNormalization())
                self.model.add(Activation('relu'))
                if dropout:
                    self.model.add(Dropout(dropout))

            self.model.add(Dense(self._feats.num_classes, activation='softmax'))

        # initiate adam optimizer
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.0, amsgrad=False)
        # Let's train the model using RMSprop
        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        self.model.summary()

    def train_test_val(self, batch_size=2, train_epochs=20, show_plots=True):

        print('Training Model:')
        # Train Model

        history = self.model.fit(self._feats.x_train, self._feats.y_train,
                                 batch_size=batch_size,
                                 epochs=train_epochs,
                                 validation_data=(self._feats.x_val, self._feats.y_val),
                                 shuffle=True,
                                 verbose=True,
                                 class_weight=self._feats.class_weights
                                 )

        # list all data in history
        print(history.history.keys())

        if show_plots:
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.semilogy(history.history['loss'])
            plt.semilogy(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

        # Test on left out Test data
        score, acc = self.model.evaluate(self._feats.x_test, self._feats.y_test,
                                         batch_size=batch_size)
        print(self.model.metrics_names)
        print('Test loss:', score)
        print('Test accuracy:', acc)

        # Build a dictionary of data to return
        data = {}
        data['score'] = score
        data['acc'] = acc

        return data
