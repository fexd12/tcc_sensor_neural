from ..utils.feats import Feats

from .model import ModelEEG

if __name__ == '__main__':
    # instancia a classe que faz o gerenciamento do model (features)
    feats = Feats(
        model_type='CNN',
    )

    # realiza a captura dos dados e
    # faz o pre processamento dos dados, aplicando fft em cada frequencia obtida anteriormente
    # monta o modelo, e cria suas features
    model = ModelEEG(feats, "/Users/felipe/Desktop/projetos/tcc/tcc/rede_neural/images/", 57600)

    # comeca o treinamento do model com os dados obtidos
    model.train_test_val()
