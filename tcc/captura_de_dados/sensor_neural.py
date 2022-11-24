import serial
from .brain.brain import Brain
from serial.serialutil import SerialException
import ctypes

class SensorNeural(serial.Serial):
    __DEFAULT_SERIAL_PORT = "/dev/ttyS0"
    __DEFAULT_SERIAL_BAUDRATE = 57600

    def __init__(self, port: str = __DEFAULT_SERIAL_PORT,
                 baudrate: int = __DEFAULT_SERIAL_BAUDRATE) -> None:
        super(SensorNeural, self).__init__(
            port=port,
            baudrate=baudrate
        )
        if not self.isOpen():
            self.open()
        self._brain = Brain()

    def capture(self):
        try:
            c_uint32 = ctypes.c_uint32 * 40

            payload = int.from_bytes(self.read(), "big")
            if self.isOpen() and self._brain.update(payload):
                # call c function update.
                powerData = self._brain.readCSV()
                if powerData is None:
                    return None
                c_uint32 = c_uint32.from_address(int(powerData))
                csv = ''
                for x in range(8):
                    # print("oi")
                    # print(c_uint32[x])
                    csv += str(c_uint32[x]) + ','
                return csv
            return None
        except SerialException:
            pass
        except Exception as e:
            print(e)
            exit()
