import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
from tcc.utils.feats import Feats
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

        if self._feats.model_type == 'CNN':
            if nunits < 2:
                print('Warning: Need at least two layers for CNN')

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(self._feats.input_shape))

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
