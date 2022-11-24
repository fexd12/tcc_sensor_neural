import os
import pandas as pd
import matplotlib.pyplot as plt

header = ["Delta", "theta", "low alpha", "high alpha", "low beta", "high beta", "low gamma", "mid gamma", "null"]


def converter_csv_image(filename):
    for i, image_path in enumerate(os.listdir(filename)):
        df = pd.read_csv(filename+image_path, sep=',', names=header)
        columns = df.columns
        # drop last column
        df.drop(columns=columns[-1], inplace=True)
        plt.plot(df)
        # plt.show()
        plt.savefig("0_rasp_" + str(i) + ".jpg")
        # print(df.head())


converter_csv_image("/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/testes/dataset/rasp_/")
