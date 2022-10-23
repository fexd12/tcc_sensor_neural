import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed

from rede_neural.feats import Feats
from rede_neural.pre_processamento import PreProcessamento

seed(1017)
set_random_seed(1017)

pd.options.display.precision = 4
plt.rcParams["figure.figsize"] = (12, 12)


class ModelEEG(PreProcessamento):
    def __init__(self, feats: Feats, filename: str, sfreq: int) -> None:
        PreProcessamento.__init__(self, feats, filename, sfreq)
        self.model = tf.keras.models.Sequential()
        self.__create_model()

    def __create_model(self, units=[16, 8, 4, 8, 16],
                       dropout=.25, batch_norm=True,
                       pool_size=2, filt_size=3):

        print('Creating ' + self._feats.model_type + ' Model')
        print('Input shape: ' + str(self._feats.input_shape))

        nunits = len(units)

        # ---DenseFeedforward Network
        # Makes a hidden layer for each item in units
        if self._feats.model_type == 'NN':
            self.model.add(tf.keras.layers.Flatten(input_shape=self._feats.input_shape))

            for unit in units:
                self.model.add(tf.keras.layers.Dense(unit))
                if batch_norm:
                    self.model.add(tf.keras.layers.BatchNormalization())
                self.model.add(tf.keras.layers.Activation('relu'))
                if dropout:
                    self.model.add(tf.keras.layers.Dropout(dropout))

            self.model.add(tf.keras.layers.Dense(self._feats.num_classes, activation='softmax'))

        if self._feats.model_type == 'CNN':
            if nunits < 2:
                print('Warning: Need at least two layers for CNN')
            self.model.add(tf.keras.layers.Conv2D(units[0], filt_size,
                           input_shape=self._feats.input_shape, padding='same'))
            self.model.add(tf.keras.layers.Activation('relu'))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding='same'))

            if nunits > 2:
                for unit in units[1:-1]:
                    self.model.add(tf.keras.layers.Conv2D(unit, filt_size, padding='same'))
                    self.model.add(tf.keras.layers.Activation('relu'))
                    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding='same'))

            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(units[-1]))
            self.model.add(tf.keras.layers.Activation('relu'))
            self.model.add(tf.keras.layers.Dense(self._feats.num_classes))
            self.model.add(tf.keras.layers.Activation('softmax'))

        # initiate adam optimizer
        opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                       decay=0.0, amsgrad=False)
        # Let's train the model using RMSprop
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        self.model.summary()

    def train_test_val(self, batch_size=2, train_epochs=200, show_plots=True):

        print('Training Model:')
        # Train Model

        check_path = './ckpt/cp-{epoch:04d}.ckpt'

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)

        history = self.model.fit(self._feats.x_train, self._feats.y_train,
                                 batch_size=batch_size,
                                 epochs=train_epochs,
                                 validation_data=(self._feats.x_val, self._feats.y_val),
                                 shuffle=True,
                                 callbacks=[save_model_cb],
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
