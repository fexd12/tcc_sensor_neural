import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tcc.rede_neural.layers import ExpandLayer
from tcc.utils.feats import Feats
from tcc.rede_neural.pre_processamento import PreProcessamento

plt.rcParams["figure.figsize"] = (12, 12)
plt.axis("off")


class ModelEEG(PreProcessamento):
    def __init__(self, feats: Feats, filename: str, sfreq: int) -> None:
        PreProcessamento.__init__(self, feats, filename, sfreq)
        self.__create_model()

    def __create_model(self, units=[200, 40],
                       dropout=.25, pool_size=20, filt_size=3, steps=2, temp_layers=4):

        self._feats.input_shape = self.train_data[0].shape[1:]
        print('Creating ' + self._feats.model_type + ' Model')
        print('Input shape: ' + str(self._feats.input_shape))

        nunits = len(units)
        convs = [self._feats.input_shape[-1] // steps for _ in range(1, steps)]
        convs += [self._feats.input_shape[-1] - sum(convs) + len(convs)]

        # ---DenseFeedforward Network
        # Makes a hidden layer for each item in units
        # if self._feats.model_type == 'NN':
        #     self.model.add(tf.keras.layers.Flatten(input_shape=self._feats.input_shape))

        #     for unit in units:
        #         self.model.add(tf.keras.layers.Dense(unit))
        #         if batch_norm:
        #             self.model.add(tf.keras.layers.BatchNormalization())
        #         self.model.add(tf.keras.layers.Activation('relu'))
        #         if dropout:
        #             self.model.add(tf.keras.layers.Dropout(dropout))

        #     self.model.add(tf.keras.layers.Dense(self._feats.num_classes, activation='softmax'))

        if self._feats.model_type == 'CNN':
            if nunits < 2:
                print('Warning: Need at least two layers for CNN')

            ins = tf.keras.layers.Input(self._feats.input_shape)

            conv = ExpandLayer()(ins)

            for i, c in enumerate(convs):
                tf.keras.layers.Conv2D(units[0] // len(convs), (1, c),
                                       use_bias=False,
                                       activation=tf.keras.activations.selu,
                                       name='spatial_conv_{0}'.format(i),
                                       kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                       data_format='channels_last'
                                       )(conv)

            tf.keras.layers.BatchNormalization()(conv)
            # self.model.add(tf.keras.layers.AveragePooling2D((pool_size, 1)))
            tf.keras.layers.SpatialDropout2D(dropout)(conv)

            for i in range(temp_layers):
                tf.keras.layers.Conv2D(units[1], (24, 1),
                                       use_bias=False,
                                       activation=tf.keras.activations.selu,
                                       name='temporeal_conv_{0}'.format(i),
                                       kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                       data_format='channels_last'
                                       )(conv)

            tf.keras.layers.BatchNormalization()(conv)
            tf.keras.layers.AveragePooling2D((pool_size, 1))(conv)
            tf.keras.layers.SpatialDropout2D(dropout)(conv)

            outs = tf.keras.layers.Flatten()(conv)

            for unit in units[2:]:
                outs = tf.keras.layers.Dense(unit,
                                             activation=tf.keras.activations.selu,
                                             kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                             )(outs)
                outs = tf.keras.layers.BatchNormalization()(outs)
                outs = tf.keras.layers.Dropout(dropout)(outs)

            outs = tf.keras.layers.Dense(self._feats.num_classes,
                                         activation='softmax',
                                         name='OUT',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                         )(outs)

        self.model = tf.keras.models.Model(ins, outs)

        # initiate adam optimizer
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=self._feats.learning_rate)
        # Let's train the model using RMSprop
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=opt,
                           metrics=[
                               tf.keras.metrics.categorical_crossentropy,
                               tf.keras.metrics.categorical_accuracy
                           ])
        self.model.summary(expand_nested=True)

    def train_test_val(self, train_epochs=50, show_plots=False):

        print('Training Model:')
        # Train Model

        check_path = './models/model_train.h5'

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path,
            save_best_only=True,
            save_weights_only=False,
            verbose=True,
            monitor='val_categorical_accuracy'
        )

        plateu = tf.keras.callbacks.ReduceLROnPlateau(patience=50, factor=0.5),


        # print(self._feats.x_train)
        # x_train = np.array(self._feats.x_train)
        # y_train = np.array(self._feats.y_train)

        # x_test = np.array(self._feats.x_val)
        # y_test = np.array(self._feats.y_val)

        # print(x_train)
        # print(y_train)

        history = self.model.fit(*self.train_data,
                                 batch_size=self._feats.batch_size,
                                 epochs=self._feats.epochs,
                                 validation_split=self._feats.test_split,
                                 callbacks=[save_model_cb, plateu],
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
        # del self.model

        predictions = self.model.predict(x=self.test_data[0],
                                         batch_size=self._feats.batch_size,
                                         verbose=True)

        print('\nPrediction Accuracy: {:.2f}'.format(100 * np.mean(
            predictions.argmax(axis=-1) == self.test_data[1].argmax(axis=-1))))
        print(self.model.metrics_names)
        np.savez('./model/results.npz',
                 predictions=predictions,
                 truth=self.test_data[1])

        # print('Test loss:', score)
        # print('Test accuracy:', acc)

        # # Build a dictionary of data to return
        # data = {}
        # data['score'] = score
        # data['acc'] = acc

        # return data
