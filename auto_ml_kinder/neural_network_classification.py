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
from scikeras.wrappers import KerasRegressor, KerasClassifier
from sklearn.metrics import mean_squared_error
from auto_ml_kinder import model_training_data_prep as mtdp
from keras.callbacks import LearningRateScheduler

class NeuralNetwork_Classification():
    data:mtdp.ModelTrainingData
    model_params = None

    def __init__(self,data:mtdp.ModelTrainingData) -> None:
        self.data = data

    def get_nn_model(self,neurons,total_columns,activation,normalization,hidden_layers,dropout,dropout_rate,optimizer):
        nn = Sequential()
        nn.add(Dense(neurons, input_dim=total_columns, activation=activation))

        if normalization > 0.5:
            nn.add(BatchNormalization())

        for _ in range(hidden_layers):
            hidden_neurons = round(neurons / (_ + 1))
            nn.add(Dense(hidden_neurons, activation=activation))

            if normalization > 0.5:
                nn.add(BatchNormalization())

            if dropout == 1:
                nn.add(Dropout(rate=dropout_rate))
        
        num_classes = len(np.unique(self.data.Y))
    
        if num_classes == 2:
            print(f'Total number of classes: {num_classes} using sigmoid as activation')
            nn.add(Dense(1, activation='sigmoid'))
            nn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            print(f'Total number of classes: {num_classes} using softmax as activation')
            nn.add(Dense(num_classes, activation='softmax'))
            nn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return nn
    
    def lr_schedule(epoch, initial_lr):
        drop = 0.5
        epochs_drop = 10
        lr = initial_lr * (drop ** np.floor(epoch / epochs_drop))
        return lr

    # Create a LearningRateScheduler callback that uses the initial learning rate
    def create_lr_scheduler(self,initial_lr):
        return LearningRateScheduler(lambda epoch: self.lr_schedule(epoch, initial_lr))

    def nn_cl_bo2(self,neurons, activation, optimizer, learning_rate, batch_size, epochs, normalization, dropout, dropout_rate,hidden_layers):
        optimizerL = [
            'Adam'
            , 'RMSprop'
            , 'Adadelta'
            , 'Adagrad'
            , 'Adamax'
            , 'Nadam'
            , 'Ftrl'
            ]
        activationL = ['relu',LeakyReLU, 'selu','elu','sigmoid','softplus','softsign','tanh','exponential']
        neurons = round(neurons)
        activation = activationL[round(activation)]
        optimizer = optimizerL[round(optimizer)].lower()
        batch_size = round(batch_size)
        epochs = round(epochs)
        # layers1 = round(layers1)
        # layers2 = round(layers2)
        hidden_layers = round(hidden_layers)
        try:
            total_columns = len(self.data.X_train[0])
        except:
            total_columns = len(self.data.X_train.columns)

        self.print_things(activation, batch_size, epochs, hidden_layers, normalization, dropout, dropout_rate, neurons, optimizer, total_columns,seperator='||')

        def get_build_function():
            return self.get_nn_model(neurons,total_columns,activation,normalization,hidden_layers,dropout,dropout_rate,optimizer)
        
        lr_scheduler = self.create_lr_scheduler(learning_rate)
        accuracy_scorer = make_scorer(accuracy_score, greater_is_better=True)
        es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
        nn = KerasClassifier(build_fn=get_build_function, epochs=epochs, batch_size=batch_size, verbose=0)
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        num_classes = len(np.unique(self.data.Y))
        if(num_classes > 2):
            score = cross_val_score(nn, self.data.X_train, self.data.Y_train_neural_network, scoring=accuracy_scorer, cv=kfold, fit_params={'callbacks':[es,lr_scheduler]}).mean()
        else:
            score = cross_val_score(nn, self.data.X_train, self.data.Y_train, scoring=accuracy_scorer, cv=kfold, fit_params={'callbacks':[es,lr_scheduler]}).mean()
        # import math
        # score = round(score,1)
        return score

    def get_best_model(self,params):
        params_nn_ = params
        learning_rate = params_nn_['learning_rate']
        activationL = ['relu',LeakyReLU, 'selu','elu','sigmoid','softplus','softsign','tanh','exponential']
        activation = activationL[round(params_nn_['activation'])]
        batch_size = round(params_nn_['batch_size'])
        epochs = round(params_nn_['epochs'])
        normalization = round(params_nn_['normalization'])
        dropout = round(params_nn_['dropout'])
        dropout_rate = params_nn_['dropout_rate']
        neurons = round(params_nn_['neurons'])
        optimizerL = ['Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
        optimizer = optimizerL[round(params_nn_['optimizer'])].lower()
        hidden_layers = round(params_nn_['hidden_layers'])
        dropout = round(params_nn_['dropout'])
        
        try:
            total_columns = len(self.data.X_train[0])
        except:
            total_columns = len(self.data.X_train.columns)
        self.print_things(activation, batch_size, epochs, hidden_layers, normalization, dropout, dropout_rate, neurons, optimizer, total_columns)

        def get_build_function():
            return self.get_nn_model(neurons,total_columns,activation,normalization,hidden_layers,dropout,dropout_rate,optimizer)
        
        lr_scheduler = self.create_lr_scheduler(learning_rate)
        es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
        nn_keras_classifier = KerasClassifier(build_fn=get_build_function, epochs=epochs, batch_size=batch_size,verbose=0)

        num_classes = len(np.unique(self.data.Y))
        if(num_classes > 2):
            nn_keras_classifier.fit(self.data.X_train, self.data.Y_train_neural_network, validation_data=(self.data.X_val, self.data.Y_val_neural_network), verbose=1, callbacks=[es,lr_scheduler])
        else:
            nn_keras_classifier.fit(self.data.X_train, self.data.Y_train, validation_data=(self.data.X_val, self.data.Y_val), verbose=1, callbacks=[es,lr_scheduler])

        return nn_keras_classifier

    def print_things(self, activation, batch_size, epochs, hidden_layers, normalization, dropout, dropout_rate, neurons, optimizer, total_columns,seperator = '\n'):
        stringToPrint = f'Neurons: {neurons}{seperator}Total Columns:{total_columns}{seperator}Normalization:{normalization > .5}{seperator}Activation:{activation}{seperator}Total hidden layers:{hidden_layers}{seperator}Dropout used:{dropout == 1}{seperator}Dropout rate if used:{dropout_rate}{seperator}Optimizer:{optimizer}{seperator}Epochs:{epochs}{seperator}Batch_size:{batch_size}\n'
        print(stringToPrint)












