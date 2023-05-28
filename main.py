from tcc.rede_neural.pre_processamento import PreProcessamento
from tcc.utils.feats import Feats
from tcc.rede_neural.model import ModelEEG


if __name__ == '__main__':
    # instancia a classe que faz o gerenciamento do model (features)
    feats = Feats(
        model_type='CNN',
    )

    process_data = PreProcessamento(feats=feats,
                                    filename="/Users/felipe/Desktop/projetos/tcc/mne_ready/",
                                    # cache_filename="/Users/felipe/Desktop/projetos/tcc/cache/",
                                    )

    # realiza a captura dos dados e
    # faz o pre processamento dos dados, aplicando fft em cada frequencia obtida anteriormente
    # monta o modelo, e cria suas features
    model = ModelEEG(feats,
                     process_data=process_data,
                     )

    # comeca o treinamento do model com os dados obtidos
    # model.train_model()

    # inference the model
    model.train_test_val()
