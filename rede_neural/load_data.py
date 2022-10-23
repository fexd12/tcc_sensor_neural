import pandas as pd
import os
import numpy as np

from labels_enum import LabelsEnum
from PIL import Image


class DataLoader():
    def __init__(self, filename: str, sfreq: int) -> None:
        self.filename = filename
        self.sfreq = sfreq
        self.raw = self._read_files()

    def _read_files(self):
        label_y: list = []
        data_x: list = []

        for i, image_path in enumerate(os.listdir(self.filename)):
            label = int(image_path.split('_')[0])
            label_one_hot = [0 if i != LabelsEnum(label).value else 1
                             for i in range(len(LabelsEnum))]
            label_y.append(label_one_hot)

            # open image and convert to grayscale
            image = Image.open(f"""{self.filename + image_path}""").convert('L')
            image_numpy = np.array(image)

            image_arr = 1 - np.reshape(image_numpy, 784) / 255.0
            data_x.append(image_arr)

        data_return = {}
        data_return["x"] = data_x
        data_return["y"] = label_y

        return data_return
