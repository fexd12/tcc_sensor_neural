from feats import Feats
from model import ModelEEG

# instancia a classe que faz o gerenciamento do model (features)
feats = Feats(
    model_type='NN',
)

# realiza a captura dos dados e
# faz o pre processamento dos dados, aplicando fft em cada frequencia obtida anteriormente
# monta o modelo, e cria suas features
model = ModelEEG(feats, feats, "/opt/nas/aguardando/*.csv", 57600)

# comeca o treinamento do model com os dados obtidos
model.train_test_val()
