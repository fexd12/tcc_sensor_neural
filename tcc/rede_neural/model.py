import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import math
from tcc.utils.feats import Feats
from tcc.rede_neural.layers import ExpandLayer
from tcc.rede_neural.pre_processamento import PreProcessamento

plt.rcParams["figure.figsize"] = (12, 12)
plt.axis("off")


class ModelEEG():
    def __init__(self, feats: Feats, process_data: PreProcessamento) -> None:
        self._feats = feats
        self.data = process_data

        # self.model = self.__create_model()
        try:
            self.model = tf.keras.models.load_model(
                './model/model_v3.h5',
                # custom_objects=dict(ExpandLayer=ExpandLayer,)
                )
        except IOError:
            print('error loading model')
            self.model = self.__create_model()

    def __create_model(self, units=[100, 20, ],
                       dropout=.25, pool_size=20, filt_size=3, steps=3, temp_layers=4):

        # self._feats.x_train = np.expand_dims(self._feats.x_train, -1)
        self._feats.input_shape = self._feats.x_train[0].shape

        print('Creating ' + self._feats.model_type + ' Model')
        print('Input shape: {}, x train data: {}'.format(
            str(self._feats.input_shape), str(self._feats.x_train[0].shape)))

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

            model = tf.keras.models.Sequential()
            # model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(self._feats.input_shape))
            # model.add(ExpandLayer())

            # model.add(tf.keras.layers.Conv2D(units[0], 3,
            #                                  name='input_layer',
            #                                  input_shape=self._feats.input_shape,
            #                                  activation=tf.keras.activations.selu,
            #                                  ))
            for i, c in enumerate(convs):

                model.add(tf.keras.layers.Conv2D(units[0] // len(convs), (1, c),
                                                 use_bias=False,
                                                 activation=tf.keras.activations.relu,
                                                 #  input_shape=self._feats.input_shape,
                                                 name='spatial_conv_{0}'.format(i),
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                                 padding='same'
                                                 ))

            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.AveragePooling2D((pool_size, 1)))
            model.add(tf.keras.layers.SpatialDropout2D(dropout))

            # for i in range(temp_layers):
            #     model.add(tf.keras.layers.Conv2D(units[1], (24, 1),
            #                                      use_bias=False,
            #                                      activation=tf.keras.activations.selu,
            #                                      name='temporal_conv_{0}'.format(i),
            #                                      kernel_regularizer=tf.keras.regularizers.l2(0.1),
            #                                      data_format='channels_last',
            #                                      padding='same'
            #                                      ))

            # model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.AveragePooling2D((pool_size, 1)))
            # model.add(tf.keras.layers.SpatialDropout2D(dropout))

            model.add(tf.keras.layers.Dense(self._feats.num_classes,
                                            activation=tf.keras.activations.relu,
                                            name='OUT_NUM_CLASSES',
                                            input_shape=self._feats.y_train[0].shape,
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                            ))
            model.add(tf.keras.layers.Flatten())

            for unit in units[2:]:
                model.add(tf.keras.layers.Dense(unit,
                                                activation=tf.keras.activations.relu,
                                                kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                                ))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(dropout))

            model.add(tf.keras.layers.Dense(self._feats.num_classes,
                                            activation='softmax',
                                            name='OUT',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                            ))

        return model

    def train_model(self, show_plots=False):
        print('Training Model:')

        # Train Model

        check_path = './model/weights/'

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=True,
            monitor='val_categorical_accuracy'
        )

        log_dir = "./model/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        plateu = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      patience=5,
                                                      factor=0.2,
                                                      min_lr=0.001)

        early = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                 patience=5,
                                                 verbose=1)

        # initiate adam optimizer
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=self._feats.learning_rate)
        # Let's train the model using RMSprop
        # model.build(self._feats.input_shape)
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=opt,
                           metrics=[
                               tf.keras.metrics.categorical_crossentropy,
                               tf.keras.metrics.categorical_accuracy
                           ])

        # for x in model.layers:
        #     x.trainable = False

        self.model.summary()

        # print(self._feats.x_train)
        # print(self._feats.y_train)
        # train_dataset_x = tf.constant(self._feats.x_train)

        # train_dataset_y = tf.constant(self._feats.y_train)

        # train_dataset = tf.data.Dataset.from_tensor_slices((*self._feats.x_train, *self._feats.y_train)) \
        #     .batch(self._feats.batch_size)

        class ProcessSequence(tf.keras.utils.Sequence):
            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return math.ceil(len(self.x) / self.batch_size)

            def __getitem__(self, idx):
                low = idx * self.batch_size
                # Cap upper bound at array length; the last batch may be smaller
                # if the total number of items is not a multiple of batch size.
                high = min(low + self.batch_size, len(self.x))
                batch_x = self.x[low:high]
                batch_y = self.y[low:high]

                return np.array(batch_x), np.array(batch_y)

        # seq = ProcessSequence(self._feats.x_train, self._feats.y_train,
        #                       self._feats.batch_size)

        history = self.model.fit(
                                 self._feats.x_train,
                                 self._feats.y_train,
                                #  train_dataset_x, train_dataset_y,
                                 epochs=self._feats.epochs,
                                 batch_size=self._feats.batch_size,
                                 validation_split=self._feats.test_split,
                                 #  validation_data=(tf.convert_to_tensor(self._feats.x_test), self._feats.y_test),
                                 #  use_multiprocessing=True,
                                 callbacks=[save_model_cb, plateu, tensorboard_callback, early],
                                 #  class_weight=self._feats.class_weights
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
        self.model.save("./model/model_v3.h5")
        # Test on left out Test data
        # del self.model

    def train_test_val(self, ):

        # model = tf.saved_model.load('./model/model_v1.h5')

        # test_dataset_x = tf.constant(*self._feats.x_test)

        # test_dataset_y = tf.constant(*self._feats.y_test)

        print(self.model.summary())

        print(len(self._feats.x_test))
        print(len(self._feats.y_test))
        print(type(self._feats.x_test))
        print(type(self._feats.y_test))
        print(self._feats.x_test[0].shape)
        print(self._feats.y_test[0].shape)
        print(self._feats.y_test[0])

        predictions = self.model.predict(self._feats.x_test,
                                         batch_size=self._feats.batch_size,
                                         verbose=True)

        print('\nPrediction:', predictions)
        print('\nPrediction:', predictions.argmax(axis=-1))
        print('\nUser real Classe:', self._feats.y_test[0].argmax(axis=-1))
        print('\nUser real Classe:', np.argmax(self._feats.y_test[0]))
        # dict(zip(self.model.metrics_names, predictions))

        # print('\nPrediction Accuracy: {:.2f}'.format(100 * np.mean(
        #     predictions.argmax(axis=-1) == self._feats.y_test[0].argmax(axis=-1))))

        # np.savez('./model/results.npz',
        #          predictions=predictions,
        #          truth=self.data.test_data[1])

        # print('Test loss:', score)
        # print('Test accuracy:', acc)

        # # Build a dictionary of data to return
        # data = {}
        # data['score'] = score
        # data['acc'] = acc

        # return data
