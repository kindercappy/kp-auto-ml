import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error
from kp_auto_ml import model_training_data_prep as mtdp

def percent_variance_explained(y_true, y_pred):
    from sklearn.metrics import r2_score
    variance_explained = r2_score(y_true, y_pred)
    percent_explained = (variance_explained * 100).round(2)
    return percent_explained

class NeuralNetwork_Params():
    neurons = None

class NeuralNetwork_Regression():
    data:mtdp.ModelTrainingData
    model_params = None

    def __init__(self,data:mtdp.ModelTrainingData) -> None:
        self.data = data


    
    def nn_cl_bo2(self,neurons, activation, optimizer, learning_rate, batch_size, epochs,
                layers1, layers2, normalization, dropout, dropout_rate):
        optimizerL = [
            'Adam'
            , 'RMSprop'
            , 'Adadelta'
            , 'Adagrad'
            , 'Adamax'
            , 'Nadam'
            , 'Ftrl'
            ]
        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                    'elu', 'exponential', LeakyReLU,'relu']
        neurons = round(neurons)
        activation = activationL[round(activation)]
        optimizer = optimizerL[round(optimizer)].lower()
        batch_size = round(batch_size)
        epochs = round(epochs)
        layers1 = round(layers1)
        layers2 = round(layers2)
        try:
            total_columns = len(self.data.X_train[0])
        except:
            total_columns = len(self.data.X_train.columns)
        def nn_cl_fun():
            nn = Sequential()
            nn.add(Dense(neurons, input_dim=total_columns, activation=activation))
            if normalization > 0.5:
                nn.add(BatchNormalization())
            for i in range(layers1):
                nn.add(Dense(neurons, activation=activation))
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate, seed=123))
            for i in range(layers2):
                nn.add(Dense(neurons, activation=activation))
            nn.add(Dense(1, activation='relu'))
            nn.compile(loss='mean_squared_error', optimizer=optimizer)
            return nn
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=True)
        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=20)
        nn = KerasRegressor(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        score = cross_val_score(nn, self.data.X_train, self.data.Y_train, scoring=mse_scorer, cv=kfold, fit_params={'callbacks':[es]}).mean()
        return -score
    
    def get_best_model(self,params):
        params_nn_ = params
        learning_rate = params_nn_['learning_rate']
        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                    'elu', 'exponential', LeakyReLU,'relu']
        activation = activationL[round(params_nn_['activation'])]
        batch_size = round(params_nn_['batch_size'])
        epochs = round(params_nn_['epochs'])
        layers1 = round(params_nn_['layers1'])
        layers2 = round(params_nn_['layers2'])
        normalization = round(params_nn_['normalization'])
        dropout = round(params_nn_['dropout'])
        dropout_rate = round(params_nn_['dropout_rate'])
        neurons = round(params_nn_['neurons'])
        optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
        optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
                    'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
                    'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
                    'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
        optimizer = optimizerL[round(params_nn_['optimizer'])].lower()
        # print(params)
        try:
            total_columns = len(self.data.X_train[0])
        except:
            total_columns = len(self.data.X_train.columns)
        def best_model_nn_cl_fun():
            nn_best_model = Sequential()
            nn_best_model.add(Dense(neurons, input_dim=total_columns, activation=activation))
            if normalization > 0.5:
                nn_best_model.add(BatchNormalization())
            for i in range(layers1):
                nn_best_model.add(Dense(neurons, activation=activation))
            if dropout > 0.5:
                nn_best_model.add(Dropout(dropout_rate, seed=123))
            for i in range(layers2):
                nn_best_model.add(Dense(neurons, activation=activation))
            nn_best_model.add(Dense(1, activation='relu'))
            nn_best_model.compile(loss='mean_squared_error', optimizer=optimizer)
            nn_best_model.summary()
            print(nn_best_model.optimizer)
            return nn_best_model
        # self.print_things(activation, batch_size, epochs, layers1, normalization, dropout, dropout_rate, neurons, optimizer, total_columns)
        es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
        nn_KerasRegressor = KerasRegressor(build_fn=best_model_nn_cl_fun, epochs=epochs, batch_size=batch_size,verbose=0)
        nn_KerasRegressor.fit(self.data.X_train, self.data.Y_train, validation_data=(self.data.X_val, self.data.Y_val), verbose=1)
        return nn_KerasRegressor

    def print_things(self, activation, batch_size, epochs, layers1, normalization, dropout, dropout_rate, neurons, optimizer, total_columns):
        print(f'neurons:{neurons}')
        print(f'total_columns:{total_columns}')
        print(f'normalization:{normalization}')
        print(f'activation:{activation}')
        print(f'layers1:{layers1}')
        print(f'dropout:{dropout}')
        print(f'dropout_rate:{dropout_rate}')
        print(f'optimizer:{optimizer}')
        print(f'epochs:{epochs}')
        print(f'batch_size:{batch_size}')
