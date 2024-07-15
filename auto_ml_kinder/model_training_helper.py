from auto_ml_kinder import model_training_data_prep as dp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from auto_ml_kinder import model_training_data_prep as dp
from auto_ml_kinder import model_training_helper as mth
from auto_ml_kinder import neural_network_regression as nnr
from auto_ml_kinder import neural_network_classification as nnc
from enum import Enum
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
from bayes_opt import BayesianOptimization
from auto_ml_kinder import model_list_helper as mlh

# Filter out specific warning messages
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def get_model_and_param_regression(power:mlh.ModelPower,model_and_param:mlh.ModelAndParamRegression):
    if not isinstance(model_and_param, mlh.ModelAndParamRegression):
        raise ValueError("Invalid model_and_param type. Please provide a valid ModelAndParam enum value.")

    model = None
    param = None
    if model_and_param == mlh.ModelAndParamRegression.Linear_Regression:
        param,model = mlh.get_parameters_linear_reg()
    elif model_and_param == mlh.ModelAndParamRegression.Ridge_Regression:
        param,model = mlh.get_parameters_ridge_fit(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.Lasso_Regression:
        param,model = mlh.get_parameters_lasso_fit(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.ElasticNet_Regression:
        param,model = mlh.get_parameters_elasticnet_fit(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.SVR_Regression:
        param,model = mlh.get_parameters_svr_fit(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.DecisionTree_Regressor:
        param,model = mlh.get_parameters_decision_tree_fit_reg(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.RandomForest_Regressor:
        param,model = mlh.get_parameters_random_forest_fit_reg(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.GradientBoosting_Regressor:
        param,model = mlh.get_parameters_gradient_boosting_fit_reg(power=power)
    elif model_and_param == mlh.ModelAndParamRegression.KNeighbors_Regressor:
        param,model = mlh.get_parameters_knn_reg(power=power)

    return model,param

def get_model_and_param_classification(power:mlh.ModelPower,model_and_param:mlh.ModelAndParamClassifiction):
    if not isinstance(model_and_param, mlh.ModelAndParamClassifiction):
        raise ValueError("Invalid model_and_param type. Please provide a valid ModelAndParam enum value.")

    model = None
    param = None
    if model_and_param == mlh.ModelAndParamClassifiction.Logistic_Regression:
        param,model = mlh.get_parameters_logistic_reg()
    elif model_and_param == mlh.ModelAndParamClassifiction.Ridge_Classifiction:
        param,model = mlh.get_parameters_ridge_classifier(power=power)
    elif model_and_param == mlh.ModelAndParamClassifiction.SVC_Classification:
        param,model = mlh.get_parameters_svc_fit(power=power)
    elif model_and_param == mlh.ModelAndParamClassifiction.DecisionTree_Classifiction:
        param,model = mlh.get_parameters_decision_tree_classifier(power=power)
    elif model_and_param == mlh.ModelAndParamClassifiction.RandomForest_Classifiction:
        param,model = mlh.get_parameters_random_forest_classifier(power=power)
    elif model_and_param == mlh.ModelAndParamClassifiction.GradientBoosting_Classifiction:
        param,model = mlh.get_parameters_gradient_boosting_classifier(power=power)
    elif model_and_param == mlh.ModelAndParamClassifiction.KNeighbors_Classifiction:
        param,model = mlh.get_parameters_knn_classifier(power=power)

    return model,param

def train_test_random_search_regression(model, param_distributions, X_train, y_train, X_test, y_test, scoring='r2', cv=5):
    total_combinations = 1
    for values in param_distributions.values():
        total_combinations *= len(values)
    # Set n_iter based on the total_combinations
    n_iter = min(total_combinations, 10)

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,n_iter=n_iter, scoring=scoring, cv=cv, random_state=42)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    
    best_model.fit(X_train, y_train)
    
    test_score = best_model.score(X_test, y_test)
    
    return best_params, best_model, test_score

def train_test_random_search_classification(model, param_distributions, X_train, y_train, X_test, y_test, scoring='accuracy', cv=5):
    total_combinations = 1
    for values in param_distributions.values():
        total_combinations *= len(values)
    
    # Set n_iter based on the total_combinations
    n_iter = min(total_combinations, 10)

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, scoring=scoring, cv=cv, random_state=42)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    
    best_model.fit(X_train, y_train)
    
    test_score = best_model.score(X_test, y_test)
    
    return best_params, best_model, test_score

class ModelPerformance():
    score = None
    model_name = None
    RMSE = None
    total_columns = None
    scaler_type = None
    selected_features = None
    num_of_clusters_if_used = None
    accuracy = None
    f1_score = None

    def __init__(self,score,model_name,RMSE = None,total_columns = 0,scaler_type = None,selected_features = None,num_of_clusters = None, accuracy = None, f1_score = None) -> None:
        self.model_name = model_name
        self.score = score
        self.RMSE = RMSE
        self.total_columns = total_columns
        self.scaler_type = scaler_type
        self.selected_features = selected_features
        self.num_of_clusters_if_used = num_of_clusters
        self.accuracy = accuracy
        self.f1_score = f1_score

    def to_dict(self):
        return vars(self)
        
def insert_object_columns(df, obj:ModelPerformance):
    if df is None:
        df = pd.DataFrame(columns=obj.to_dict())
    
    row_data = obj.to_dict()
    
    df_new = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    
    return df_new


class ModelMeta():
    model = None
    model_params = None
    
    def __init__(self,model,params) -> None:
        self.model = model
        self.model_params = params

class NeuralNetwork_BayesianOptimization():
    nn_maximised:BayesianOptimization = None
    nn_regressor:nnr.NeuralNetwork_Regression = None
    nn_classifier:nnc.NeuralNetwork_Classification = None

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
        

class ModelTrainer():
    performance_df:pd.DataFrame = None
    data:dp.ModelTrainingData = None
    models:list[ModelMeta] = []
    neural_network_bayesian_optimization:NeuralNetwork_BayesianOptimization = None
    def __init__(self,data:dp.ModelTrainingData) -> None:
        self.data = data
        self.neural_network_bayesian_optimization = NeuralNetwork_BayesianOptimization()

    def perform_operation_regression(self, permutate_n_less_column = 0, exclude_models: list[mlh.ModelAndParamRegression] = []):
        for model_and_param in mlh.ModelAndParamRegression:
            skip_this_model = any(exclude_model.name == model_and_param.name for exclude_model in exclude_models)
            if(skip_this_model):
                continue
            model,param =get_model_and_param_regression(power=mlh.ModelPower.HIGH,model_and_param=model_and_param)
            for X_train, X_val, X_test,selected_columns in self.data.generate_permutations_train(min_columns=len(self.data.X_original.columns)-permutate_n_less_column):
                best_param,best_model,score = train_test_random_search_regression(model=model,param_distributions=param,X_train=X_train,y_train=self.data.Y_train,X_test=X_val,y_test=self.data.Y_val)
                self.predictor_regression(model_and_param.name, best_param, best_model,X_test,selected_columns=selected_columns)

    def perform_operation_classification(self, permutate_n_less_column = 0, exclude_models: list[mlh.ModelAndParamClassifiction] = []):
        for model_and_param in mlh.ModelAndParamClassifiction:
            skip_this_model = any(exclude_model.name == model_and_param.name for exclude_model in exclude_models)
            if(skip_this_model):
                continue
            model,param = get_model_and_param_classification(power=mlh.ModelPower.HIGH,model_and_param=model_and_param)
            for X_train, X_val, X_test,selected_columns in self.data.generate_permutations_train(min_columns=len(self.data.X_original.columns)-permutate_n_less_column):
                best_param,best_model,score = train_test_random_search_classification(model=model,param_distributions=param,X_train=X_train,y_train=self.data.Y_train,X_test=X_val,y_test=self.data.Y_val)
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

        num_of_clusters = extract_last_digit_from_list(selected_columns, 'cluster')
        
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
        self.performance_df = insert_object_columns(self.performance_df, model_performance)

    def predictor_regression(self, model_name, best_param, best_model,X_test,selected_columns = []):
        from sklearn.metrics import mean_squared_error
        try:
            total_columns = len(X_test[0])
        except:
            total_columns = len(X_test.columns)
            
        y_pred = best_model.predict(X_test)
        test_score = best_model.score(X_test, self.data.Y_test)

        rmse = mean_squared_error(self.data.Y_test, y_pred, squared=False)
        num_of_clusters = extract_last_digit_from_list(selected_columns,'cluster')
        model_performance = ModelPerformance(score=test_score,model_name=model_name,RMSE=rmse,total_columns=total_columns,scaler_type=type(self.data.Normalizer).__name__,selected_features=', '.join(selected_columns),num_of_clusters=num_of_clusters)
        self.models.append(ModelMeta(best_model,best_param))
        self.performance_df = insert_object_columns(self.performance_df,model_performance)

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
        self.neural_network_bayesian_optimization.nn_regressor =  nnr.NeuralNetwork_Regression(self.data)
        
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
        self.neural_network_bayesian_optimization.nn_classifier =  nnc.NeuralNetwork_Classification(self.data)
        
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
        y_pred_int = np.argmax(y_pred, axis=1).astype(np.float32)
        y_test_int = np.argmax(self.data.Y_test_neural_network, axis=1).astype(np.float32)
        test_score = best_model.score(X_test, self.data.Y_test_neural_network)
        accuracy = accuracy_score(y_test_int, y_pred_int)
        f1 = f1_score(y_test_int, y_pred_int, average='weighted')

        num_of_clusters = extract_last_digit_from_list(selected_columns, 'cluster')
        
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
        self.performance_df = insert_object_columns(self.performance_df, model_performance)
    
def extract_last_digit_from_list(select_columns, match_string):
    # Initialize last_digit as None
    last_digit = None

    # Iterate through each column name in select_columns
    for column in select_columns:
        # Check if match_string is present in the column name
        if match_string in column:
            # Extract the last digit from the matched column name
            last_digit = column.split(match_string)[-1][-1]
            break  # Exit the loop after finding the first match

    return 0 if last_digit == None else int(last_digit)