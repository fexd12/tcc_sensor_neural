import serial
from serial.serialutil import SerialException
from brain.brain import Brain


class SensorNeural(serial.Serial, Brain):
    __DEFAULT_SERIAL_PORT = "/dev/ttyS0"
    __DEFAULT_SERIAL_BAUDRATE = "57600"

    def __init__(self, port: str, baudrate: str) -> None:
        super(SensorNeural, self).__init__(
            port=(port or self.__DEFAULT_SERIAL_PORT),
            baudrate=(baudrate or self.__DEFAULT_SERIAL_BAUDRATE)
        )

    def capture(self) -> str:
        try:
            if self.is_open() and self.update(self.read()):
                # call c function update.
                return self.readCSV()
        except SerialException:
            pass
        except Exception as e:
            print(e)
