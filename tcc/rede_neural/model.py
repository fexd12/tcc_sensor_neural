import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from ..utils.feats import Feats
from .pre_processamento import PreProcessamento

plt.rcParams["figure.figsize"] = (12, 12)
plt.axis("off")


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
            print(units)
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
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        self.model.summary()

    def train_test_val(self, batch_size=2, train_epochs=50, show_plots=False):

        print('Training Model:')
        # Train Model

        check_path = './checkpoints/cp-{epoch:04d}.ckpt'

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, save_freq=5)

        # print(self._feats.x_train)
        x_train = np.array(self._feats.x_train)
        y_train = np.array(self._feats.y_train)

        x_test = np.array(self._feats.x_val)
        y_test = np.array(self._feats.y_val)

        # print(x_train)
        # print(y_train)

        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=train_epochs,
                                 validation_data=(x_test, y_test),
                                 shuffle=True,
                                 callbacks=[save_model_cb],
                                 #  class_weight=np.asarray(self._feats.class_weights)
                                 )

        # list all data in history
        print(history.history.keys())

        if show_plots:
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
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

        print("saving model...")
        self.model.save("./model/model_v1.h5")
        # Test on left out Test data
        score, acc = self.model.evaluate(x_test, y_test,
                                         batch_size=batch_size)

        print(self.model.metrics_names)
        print('Test loss:', score)
        print('Test accuracy:', acc)

        # Build a dictionary of data to return
        data = {}
        data['score'] = score
        data['acc'] = acc

        return data
