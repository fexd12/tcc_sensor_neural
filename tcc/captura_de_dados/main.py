import datetime
import sys
from time import time

from .sensor_neural import SensorNeural

if __name__ == "__main__":
    try:
        file_date = datetime.datetime.now().timestamp()
        sensor_leitura = SensorNeural()
        while True:
            # TO DO - colocar o numero do enum correspondente
            pensamento = input("pensamento a processar")
            time_inicial = time()
            with open(f"{pensamento}_{file_date}.csv", "a") as f:
                while True:
                    captura = sensor_leitura.capture()
                    if captura is None:
                        continue
                    f.writelines(captura + '\n')

                    # 1 minuto em segundos
                    if round((time() - time_inicial), 2) > 60:
                        break
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(e)
