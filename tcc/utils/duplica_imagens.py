import os
from PIL import Image


class ImageRotate():
    def __init__(self, filename: str, angle: int):
        self.filename = filename
        self.angle = angle
        self.read_files()

    def read_files(self):
        for i, image_path in enumerate(os.listdir(self.filename)):
            image_file_input = self.filename + image_path
            image_file_split = image_file_input.split('.')

            image = Image.open(image_file_input).convert('L')
            image = image.resize(size=(1000, 350))
            image_rotate: Image.Image = image.rotate(self.angle)

            image_file_output = "/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/" + \
                image_path

            image_file_split_output = image_file_output.split('.')

            image_rotate.save(
                image_file_split_output[0] +
                '-' +
                str(self.angle) +
                '.' +
                image_file_split[1]
            )


ImageRotate("/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/processar/", 90)
ImageRotate("/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/processar/", 180)
ImageRotate("/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/processar/", 270)
ImageRotate("/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/processar/", 0)
