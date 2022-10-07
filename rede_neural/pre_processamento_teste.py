from pre_processamento import PreProcessamento
from feats import Feats

feats_ = Feats()

pre = PreProcessamento(feats_, 'testes/dataset/*.csv', 57600)

print(feats_.x_train)
