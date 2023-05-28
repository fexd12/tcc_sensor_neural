import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)


class Motor():
    def __init__(self, pin):
        self.pin = pin
        GPIO.setup(self.pin, GPIO.OUT)

    # def movimento(self, value: LabelsEnum):
    #     if value == LabelsEnum.UP:
    #         GPIO.output(self.pin, GPIO.HIGH)

    def activate(self):
        GPIO.output(self.pin, GPIO.HIGH)

    def deactivate(self):
        GPIO.output(self.pin, GPIO.LOW)
