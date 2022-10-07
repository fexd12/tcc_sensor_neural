import sys
import datetime
from time import time
from sensor_neural import SensorNeural

if __name__ == "__main__":
    try:
        file_date = datetime.datetime.now().timestamp()
        sensor_leitura = SensorNeural()
        while True:
            pensamento = input("pensamento a processar")
            time_inicial = time()
            with open(f"/opt/nas/aguardando/{pensamento}_{file_date}.csv", "a") as f:
                while True:
                    captura = sensor_leitura.capture()
                    f.writelines(captura)

                    # 1 minuto em segundos
                    if round((time() - time_inicial), 2) > 60:
                        break
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(e)
