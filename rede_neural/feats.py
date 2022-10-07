class Feats:
    def __init__(self, num_classes=2, class_weights: list = [1, 1], input_shape: list = [16, ],
                 new_times=1, model_type='1', x_train: list = [], y_train: list = [], x_test: list = [],
                 y_test: list = [], x_val: list = [], y_val: list = [], test_split=0.2,
                 val_split=0.2, random_seed=1017):

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
        self.val_split = val_split
        self.random_seed = random_seed
