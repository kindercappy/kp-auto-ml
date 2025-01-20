from bayes_opt import BayesianOptimization
from auto_ml_kinder.classes.NeuralNetwork_Regression import NeuralNetwork_Regression
from auto_ml_kinder.classes.NeuralNetwork_Classification import NeuralNetwork_Classification

class NeuralNetwork_BayesianOptimization():
    nn_maximised:BayesianOptimization = None
    nn_regressor:NeuralNetwork_Regression = None
    nn_classifier:NeuralNetwork_Classification = None