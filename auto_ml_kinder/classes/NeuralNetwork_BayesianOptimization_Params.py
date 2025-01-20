class NeuralNetwork_BayesianOptimization_Params():
    neurons_min_max = None
    activation_min_max = None
    optimizer_min_max = None
    learning_rate_min_max = None
    batch_size_min_max = None
    epochs_min_max = None
    normalization_min_max = None
    dropout_rate_min_max = None
    dropout_min_max = None
    hidden_layers_min_max = None

    def __init__(self
                 ,neurons_min_max = (32, 128)
                 ,learning_rate_min_max = (0.001, .01)
                 ,batch_size_min_max = (32, 64)
                 ,epochs_min_max = (50, 100)
                 ,normalization_min_max = (0,1)
                 ,dropout_rate_min_max = (0.2,0.6)
                 ,hidden_layers_min_max = (1,2)
                 ,dropout_min_max = (0,1)
                 ,activation_min_max = (0, 9) ):
        self.neurons_min_max = neurons_min_max
        self.activation_min_max = activation_min_max
        self.optimizer_min_max = (0,6)
        self.learning_rate_min_max = learning_rate_min_max
        self.batch_size_min_max = batch_size_min_max
        self.epochs_min_max = epochs_min_max
        self.normalization_min_max = normalization_min_max
        self.dropout_min_max = dropout_min_max
        self.dropout_rate_min_max = dropout_rate_min_max
        self.hidden_layers_min_max = hidden_layers_min_max
  