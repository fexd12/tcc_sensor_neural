import RPi.GPIO as GPIO
from ..utils.labels_enum import LabelsEnum

GPIO.setmode(GPIO.BOARD)


class Motor():
    def __init__(self, pin):
        self.pin = pin
        GPIO.setup(self.pin, GPIO.OUT)

    def movimento(self, value: LabelsEnum):
        if value == LabelsEnum.UP:
            GPIO.output(self.pin, GPIO.HIGH)
