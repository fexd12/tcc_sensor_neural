import os
import numpy as np

from PIL import Image

from tcc.utils.feats import Feats
from ..utils.labels_enum import LabelsEnum


class DataLoader():
    def __init__(self, filename: str, sfreq: int, feats: Feats) -> None:
        self.filename = filename
        self.sfreq = sfreq
        self._feats = feats
        self.raw = self._read_files()

    def _read_files(self):
        label_y: list = []
        data_x = []

        for i, image_path in enumerate(os.listdir(self.filename)):
            label = int(image_path.split('_')[0])
            label_one_hot = [0 if i != LabelsEnum(label).value else 1
                             for i in range(len(list(LabelsEnum)))]
            label_y.append(label_one_hot)

            # open image and convert to grayscale
            image_file = self.filename + image_path
            # print(image_file)
            # print(image_file)
            image = Image.open(image_file).convert('L')
            image_numpy = np.array(image).reshape(self._feats.input_shape)
            # print("img shape: " + str(image_numpy.shape))

            image_arr = 1 - image_numpy / 255.0
            data_x.append(image_arr)
            # print("img before shape: " + str(image_numpy))
            # print("img after shape: " + str(image_arr.tolist()))
            # yield image_numpy, label_one_hot,

        data_return = {}
        # print(data_x)
        data_return["x"] = data_x
        data_return["y"] = label_y

        return data_return
