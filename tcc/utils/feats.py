from tcc.utils.labels_enum import DataEnum


class Feats:
    def __init__(self, num_classes=len(DataEnum), class_weights: list = [1, 1], input_shape: tuple = (288, 1125, 26),
                 new_times=1, model_type='CNN', x_train=[], y_train=[], x_test=[],
                 y_test=[], x_val=[], y_val=[], test_split=0.3, batch_size=10,
                 val_split=0.2, random_seed=1017, learning_rate=0.00001, epochs=100,
                 t_min=-0.5, t_len=4.5):

        self.num_classes = num_classes
        self.class_weights = class_weights
        self.input_shape = input_shape
        self.new_times = new_times
        self.model_type = model_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.test_split = test_split
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.t_min = t_min
        self.t_len = t_len
