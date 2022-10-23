import serial
from brain.brain import Brain
from serial.serialutil import SerialException


class SensorNeural(serial.Serial, Brain):
    __DEFAULT_SERIAL_PORT = "/dev/ttyS0"
    __DEFAULT_SERIAL_BAUDRATE = 57600

    def __init__(self, port: str = __DEFAULT_SERIAL_PORT,
                 baudrate: int = __DEFAULT_SERIAL_BAUDRATE) -> None:
        super(SensorNeural, self).__init__(
            port=port,
            baudrate=baudrate
        )

    def capture(self) -> str:
        try:
            if self.isOpen() and self.update(self.read()):
                # call c function update.
                return self.readCSV()
        except SerialException:
            pass
        except Exception as e:
            print(e)
