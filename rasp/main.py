import tensorflow as tf
import time
from tcc.utils.labels_enum import DataEnum
from tcc.utils.feats import Feats
from tcc.motor import Motor
from tcc.rede_neural.pre_processamento import PreProcessamento

# plt.rcParams["figure.figsize"] = (12, 12)
# plt.axis("off")

# leitura do model treinado
try:
    model = tf.keras.models.load_model(
        '/home/ubuntu/projetos/tcc2/model/model_v3.h5',
        # custom_objects=dict(ExpandLayer=ExpandLayer,)
        )
    print(model.summary())
except IOError:
    print('error loading model')

# GPIO 4
motor_esquerda = Motor(4)

# GPIO 17
motor_direita = Motor(17)

_feats = Feats()
process_data = PreProcessamento(feats=_feats,
                                filename="/home/ubuntu/projetos/tcc2/mne_ready/",
                                # cache_filename="/Users/felipe/Desktop/projetos/tcc/cache/",
                                )

while True:
    print('predicting results...')
    predictions = model.predict(_feats.x_test,
                                batch_size=_feats.batch_size,
                                verbose=True)

    classes = predictions.argmax(axis=-1)
    print('\nPrediction:', classes)

    predict = DataEnum(classes)

    if predict == DataEnum.LEFT_HAND:
        print('forward prediction')
        motor_esquerda.activate()
        motor_direita.activate()
    elif predict == DataEnum.RIGHT_HAND:
        print('Right hand prediction')
        motor_esquerda.activate()
        motor_direita.activate()

    # if predict == DataEnum.UP:
    #     motor_esquerda.activate()
    #     motor_direita.activate()
    # elif predict == DataEnum.DOWN:
    #     pass

    time.sleep(4)
    motor_esquerda.deactivate()
    motor_direita.deactivate()
    # GPIO.output(pin_1, GPIO.HIGH)
    break
