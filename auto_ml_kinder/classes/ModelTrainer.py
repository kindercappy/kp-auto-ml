from auto_ml_kinder.classes.Functions import Functions
import pandas as pd
from auto_ml_kinder.classes.ModelMeta import ModelMeta
from auto_ml_kinder.classes.NeuralNetwork_BayesianOptimization import NeuralNetwork_BayesianOptimization
from auto_ml_kinder.classes.ModelAndParamRegression import ModelAndParamRegression
from auto_ml_kinder.classes.ModelPower import ModelPower
from auto_ml_kinder.classes.ModelAndParamClassifiction import ModelAndParamClassifiction
from auto_ml_kinder.classes.ModelPerformance import ModelPerformance
from auto_ml_kinder.classes.NeuralNetwork_BayesianOptimization_Params import NeuralNetwork_BayesianOptimization_Params
from auto_ml_kinder.classes.NeuralNetwork_Regression import NeuralNetwork_Regression
from auto_ml_kinder.classes.NeuralNetwork_Classification import NeuralNetwork_Classification
from auto_ml_kinder.classes.ModelTrainingData import ModelTrainingData
from bayes_opt import BayesianOptimization

class ModelTrainer():
    performance_df:pd.DataFrame = None
    data:ModelTrainingData = None
    models:list[ModelMeta] = []
    neural_network_bayesian_optimization:NeuralNetwork_BayesianOptimization = None
    func = None
    def __init__(self,data:ModelTrainingData) -> None:
        self.data = data
        self.neural_network_bayesian_optimization = NeuralNetwork_BayesianOptimization()
        self.func = Functions()

    def perform_operation_regression(self, permutate_n_less_column = 0, exclude_models: list[ModelAndParamRegression] = []):
        for model_and_param in ModelAndParamRegression:
            skip_this_model = any(exclude_model.name == model_and_param.name for exclude_model in exclude_models)
            if(skip_this_model):
                continue
            model,param = self.func.get_model_and_param_regression(power=ModelPower.HIGH,model_and_param=model_and_param)
            for X_train, X_val, X_test,selected_columns in self.data.generate_permutations_train(min_columns=len(self.data.X_original.columns)-permutate_n_less_column):
                best_param,best_model,score = self.func.train_test_random_search_regression(model=model,param_distributions=param,X_train=X_train,y_train=self.data.Y_train,X_test=X_val,y_test=self.data.Y_val)
                self.predictor_regression(model_and_param.name, best_param, best_model,X_test,selected_columns=selected_columns)

    def perform_operation_classification(self, permutate_n_less_column = 0, exclude_models: list[ModelAndParamClassifiction] = []):
        for model_and_param in ModelAndParamClassifiction:
            skip_this_model = any(exclude_model.name == model_and_param.name for exclude_model in exclude_models)
            if(skip_this_model):
                continue
            model,param = self.func.get_model_and_param_classification(power=ModelPower.HIGH,model_and_param=model_and_param)
            for X_train, X_val, X_test,selected_columns in self.data.generate_permutations_train(min_columns=len(self.data.X_original.columns)-permutate_n_less_column):
                best_param,best_model,score = self.func.train_test_random_search_classification(model=model,param_distributions=param,X_train=X_train,y_train=self.data.Y_train,X_test=X_val,y_test=self.data.Y_val)
                self.predictor_classification(model_and_param.name, best_param, best_model,X_test,selected_columns=selected_columns)

    def predictor_classification(self, model_name, best_param, best_model, X_test, selected_columns=[]):
        from sklearn.metrics import accuracy_score, f1_score
        import numpy as np
        try:
            total_columns = len(X_test[0])
        except:
            total_columns = len(X_test.columns)
        
        y_pred = best_model.predict(X_test)
        test_score = best_model.score(X_test, self.data.Y_test)
        accuracy = accuracy_score(self.data.Y_test, y_pred)
        f1 = f1_score(self.data.Y_test, y_pred, average='weighted')

        num_of_clusters = self.func.extract_last_digit_from_list(selected_columns, 'cluster')
        
        model_performance = ModelPerformance(
            score=test_score,
            model_name=model_name,
            accuracy=accuracy,
            f1_score=f1,
            total_columns=total_columns,
            scaler_type=type(self.data.Normalizer).__name__,
            selected_features=', '.join(selected_columns),
            num_of_clusters=num_of_clusters
        )
        
        self.models.append(ModelMeta(best_model, best_param))
        self.performance_df = self.func.insert_object_columns(self.performance_df, model_performance)

    def predictor_regression(self, model_name, best_param, best_model,X_test,selected_columns = []):
        from sklearn.metrics import mean_squared_error
        try:
            total_columns = len(X_test[0])
        except:
            total_columns = len(X_test.columns)
            
        y_pred = best_model.predict(X_test)
        test_score = best_model.score(X_test, self.data.Y_test)

        rmse = mean_squared_error(self.data.Y_test, y_pred, squared=False)
        num_of_clusters = self.func.extract_last_digit_from_list(selected_columns,'cluster')
        model_performance = ModelPerformance(score=test_score,model_name=model_name,RMSE=rmse,total_columns=total_columns,scaler_type=type(self.data.Normalizer).__name__,selected_features=', '.join(selected_columns),num_of_clusters=num_of_clusters)
        self.models.append(ModelMeta(best_model,best_param))
        self.performance_df = self.func.insert_object_columns(self.performance_df,model_performance)

    def predict_test_data(self,X,model):
        import numpy as np
        X_result = None

        if(self.data.create_clustering_feature_and_no_of_clusters[0]):
            labels = self.data.only_predict(X,self.data.clustering_model)
            if isinstance(X, pd.DataFrame):
                X_result = X.copy()
                X_result[f'cluster_{self.data.create_clustering_feature_and_no_of_clusters[1]}'] = labels
            elif isinstance(X, np.ndarray):
                X_result = np.hstack((X, labels.reshape(-1, 1)))
            elif isinstance(X, list):
                X_result = [row + [label] for row, label in zip(X, labels)]
        
        X_result = self.data.Data_transformer_pipe.transform(X_result)

        predictions = model.predict(X_result)

        return predictions

    def perform_neural_network_regression(self
                                          ,totalExperiments = 4
                                          ,params:NeuralNetwork_BayesianOptimization_Params = NeuralNetwork_BayesianOptimization_Params(
                                              neurons_min_max= (32, 128),
                                              batch_size_min_max=(32, 64),
                                              dropout_rate_min_max=(0.2,0.6),
                                              epochs_min_max=(50, 100),
                                              hidden_layers_min_max=(1,6),
                                              learning_rate_min_max=(0.001, .01),
                                              normalization_min_max=(0,1)
                                          )):
        PARAMS = {
            'neurons': params.neurons_min_max,
            'activation':params.activation_min_max,
            'optimizer':params.optimizer_min_max,
            'learning_rate':params.learning_rate_min_max,
            'batch_size':params.batch_size_min_max,
            'epochs':params.epochs_min_max,
            'normalization':params.normalization_min_max,
            'dropout':params.dropout_min_max,
            'dropout_rate':params.dropout_rate_min_max,
            'hidden_layers':params.hidden_layers_min_max
        }

        self.neural_network_bayesian_optimization = NeuralNetwork_BayesianOptimization()
        self.neural_network_bayesian_optimization.nn_regressor =  NeuralNetwork_Regression(self.data)
        
        # Run Bayesian Optimization
        self.neural_network_bayesian_optimization.nn_maximised = BayesianOptimization(self.neural_network_bayesian_optimization.nn_regressor.nn_cl_bo2, PARAMS, random_state=111,verbose=2)
        init,iter = int(totalExperiments / 2),int(totalExperiments / 2)
        self.neural_network_bayesian_optimization.nn_maximised.maximize(init_points=init, n_iter=iter)

    def perform_neural_network_classification(self
                                          ,totalExperiments = 4
                                          ,params:NeuralNetwork_BayesianOptimization_Params = NeuralNetwork_BayesianOptimization_Params(
                                              neurons_min_max= (32, 128),
                                              batch_size_min_max=(32, 64),
                                              dropout_rate_min_max=(0.2,0.6),
                                              epochs_min_max=(50, 100),
                                              hidden_layers_min_max=(1,6),
                                              learning_rate_min_max=(0.001, .01),
                                              normalization_min_max=(0,1)
                                          )):
        PARAMS = {
            'neurons': params.neurons_min_max,
            'activation':params.activation_min_max,
            'optimizer':params.optimizer_min_max,
            'learning_rate':params.learning_rate_min_max,
            'batch_size':params.batch_size_min_max,
            'epochs':params.epochs_min_max,
            'normalization':params.normalization_min_max,
            'dropout':params.dropout_min_max,
            'dropout_rate':params.dropout_rate_min_max,
            'hidden_layers':params.hidden_layers_min_max
        }

        self.neural_network_bayesian_optimization = NeuralNetwork_BayesianOptimization()
        self.neural_network_bayesian_optimization.nn_classifier =  NeuralNetwork_Classification(self.data)
        
        # Run Bayesian Optimization
        self.neural_network_bayesian_optimization.nn_maximised = BayesianOptimization(self.neural_network_bayesian_optimization.nn_classifier.nn_cl_bo2, PARAMS, random_state=111,verbose=2)
        init,iter = int(totalExperiments / 2),int(totalExperiments / 2)
        self.neural_network_bayesian_optimization.nn_maximised.maximize(init_points=init, n_iter=iter)


    def neural_network_best_model_regression(self,epochs = None):
        params_nn = self.neural_network_bayesian_optimization.nn_maximised.max['params']
        if epochs is not None:
            params_nn['epochs'] = epochs
        predictor = self.neural_network_bayesian_optimization.nn_regressor.get_best_model(params_nn)
        predictor.fit(self.data.X_train, self.data.Y_train)
        self.predictor_regression(model_name='NeuralNetwork',best_model=predictor,best_param=predictor.get_params(),X_test=self.data.X_test,selected_columns=self.data.Selected_Features)

    def neural_network_best_model_classification(self,epochs = None):
        import numpy as np
        import keras as k
        params_nn = self.neural_network_bayesian_optimization.nn_maximised.max['params']
        if epochs is not None:
            params_nn['epochs'] = epochs
        predictor = self.neural_network_bayesian_optimization.nn_classifier.get_best_model(params_nn)
        num_classes = len(np.unique(self.data.Y))
        if num_classes > 2:
            predictor.fit(self.data.X_train, self.data.Y_train_neural_network)
        else:
            predictor.fit(self.data.X_train, self.data.Y_train)
        self.predictor_neural_network_classification(model_name='NeuralNetwork',best_model=predictor,best_param=predictor.get_params(),X_test=self.data.X_test,selected_columns=self.data.Selected_Features)

    def predictor_neural_network_classification(self, model_name, best_param, best_model, X_test, selected_columns=[]):
        from sklearn.metrics import accuracy_score, f1_score
        import numpy as np
        try:
            total_columns = len(X_test[0])
        except:
            total_columns = len(X_test.columns)
        
        y_pred = best_model.predict(X_test)
        if y_pred.ndim == 1:# because we have only single neuron in last layer then prediction will be single value
            y_pred_int = y_pred
        else:# we have more than 1 neuron in last layer then prediction will be multiple then get max value
            y_pred_int = np.argmax(y_pred, axis=1).astype(np.float32)
        
        if y_pred.ndim == 1:# because we have only single neuron in last layer then prediction will be single value
            y_test_int = self.data.Y_test_neural_network
        else:# we have more than 1 neuron in last layer then prediction will be multiple then get max value
            y_test_int = np.argmax(self.data.Y_test_neural_network, axis=1).astype(np.float32)

        test_score = best_model.score(X_test, self.data.Y_test_neural_network)
        accuracy = accuracy_score(y_test_int, y_pred_int)
        f1 = f1_score(y_test_int, y_pred_int, average='weighted')

        num_of_clusters = self.func.extract_last_digit_from_list(selected_columns, 'cluster')
        
        model_performance = ModelPerformance(
            score=test_score,
            model_name=model_name,
            accuracy=accuracy,
            f1_score=f1,
            total_columns=total_columns,
            scaler_type=type(self.data.Normalizer).__name__,
            selected_features=', '.join(selected_columns),
            num_of_clusters=num_of_clusters
        )
        
        self.models.append(ModelMeta(best_model, best_param))
        self.performance_df = self.func.insert_object_columns(self.performance_df, model_performance)
  