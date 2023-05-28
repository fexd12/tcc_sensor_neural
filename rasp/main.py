import tensorflow as tf
import numpy as np
import pandas as pd
from tcc.utils import labels_enum
# from tcc.captura_de_dados import sensor_neural
from PIL import Image
from tcc.motor import Motor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas


# leitura do model treinado
model: tf.keras.Model = tf.keras.models.load_model('/Users/felipe/Desktop/projetos/tcc/model/model_v1.h5')

# sensor_leitura = sensor_neural.SensorNeural()

# GPIO 16
motor_esquerda = Motor(16, 71)

header = ["Delta", "theta", "low alpha", "high alpha", "low beta", "high beta", "low gamma", "mid gamma", "null"]

while True:
    leitura = ''
    # espera 6 seg para captura dos dados
    # i = 0
    # while True:
        # leitura_tmp = sensor_leitura.capture()
        # if leitura_tmp is None:
        #     continue
        # if i == 6:
        #     break
        # i += 1

        # leitura += leitura_tmp + '\n'

    # usar input do dataset 
    df = pd.read_csv(leitura, sep=',')
    columns = df.columns
    # drop last column
    df.drop(columns=columns[-1], inplace=True)

    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)

    plt.axis('off')
    canvas.draw()
    X = np.array(canvas.renderer.buffer_rgba())

    # image = Image.open("/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/0_rasp_0-0.jpg")
    image = Image.fromarray(X, mode="L")
    image = image.resize(size=(1000, 350))
    image.save("teste_rasp_0_" + np.random.random)

    image_numpy = np.array(image)
    image_arr = 1 - image_numpy / 255.0

    matriz = model.predict(image_numpy.reshape((-1, 350, 1000, 1)))

    classes = np.argmax(matriz)

    predict = labels_enum.LabelsEnum(classes)

    # motor.movimento(predict)
    motor.movimento(predict)