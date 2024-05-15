from kp_auto_ml import model_training_data_prep as dp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from kp_auto_ml import model_training_data_prep as dp
from kp_auto_ml import model_training_helper as mth

from enum import Enum
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings

from kp_auto_ml import model_list_helper as mlh

# Filter out specific warning messages
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def get_model_and_param(power:mlh.ModelPower,model_and_param:mlh.ModelAndParam):
    if not isinstance(model_and_param, mlh.ModelAndParam):
        raise ValueError("Invalid model_and_param type. Please provide a valid ModelAndParam enum value.")

    model = None
    param = None
    if model_and_param == mlh.ModelAndParam.Linear_Regression:
        param,model = mlh.get_parameters_linear_reg()
    elif model_and_param == mlh.ModelAndParam.Ridge_Regression:
        param,model = mlh.get_parameters_ridge_fit(power=power)
    elif model_and_param == mlh.ModelAndParam.Lasso_Regression:
        param,model = mlh.get_parameters_lasso_fit(power=power)
    elif model_and_param == mlh.ModelAndParam.ElasticNet_Regression:
        param,model = mlh.get_parameters_elasticnet_fit(power=power)
    elif model_and_param == mlh.ModelAndParam.SVR_Regression:
        param,model = mlh.get_parameters_svr_fit(power=power)
    elif model_and_param == mlh.ModelAndParam.DecisionTree_Regressor:
        param,model = mlh.get_parameters_decision_tree_fit_reg(power=power)
    elif model_and_param == mlh.ModelAndParam.RandomForest_Regressor:
        param,model = mlh.get_parameters_random_forest_fit_reg(power=power)
    elif model_and_param == mlh.ModelAndParam.GradientBoosting_Regressor:
        param,model = mlh.get_parameters_gradient_boosting_fit_reg(power=power)
    elif model_and_param == mlh.ModelAndParam.KNeighbors_Regressor:
        param,model = mlh.get_parameters_knn_reg(power=power)

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


class ModelPerformance():
    score = None
    model_name = None
    RMSE = None
    def __init__(self,score,model_name,RMSE = None) -> None:
        self.model_name = model_name
        self.score = score
        self.RMSE = RMSE

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

class ModelTrainer():
    performance_df:pd.DataFrame = None
    data:dp.ModelTrainingData = None
    models:list[ModelMeta] = []

    def __init__(self,data:dp.ModelTrainingData) -> None:
        self.data = data

    def perform_operation_regression(self, permutate_n_less_column = 0, exclude_models: list[mlh.ModelAndParam] = []):
        for scaler in mlh.ModelAndParam:
            skip_this_model = any(exclude_model.name == scaler.name for exclude_model in exclude_models)
            if(skip_this_model):
                continue
            model,param =get_model_and_param(power=mlh.ModelPower.HIGH,model_and_param=scaler)
            for X_train, X_val, X_test in self.data.generate_permutations_train(min_columns=len(self.data.X_original.columns)-permutate_n_less_column):
                best_param,best_model,score = train_test_random_search_regression(model=model,param_distributions=param,X_train=X_train,y_train=self.data.Y_train,X_test=X_val,y_test=self.data.Y_val)

                y_pred = best_model.predict(X_test)
                
                from sklearn.metrics import mean_squared_error

                rmse = mean_squared_error(self.data.Y_test, y_pred, squared=False)
                model_performance = ModelPerformance(score=score,model_name=scaler.name,RMSE=rmse)
                self.models.append(ModelMeta(best_model,best_param))
                self.performance_df = insert_object_columns(self.performance_df,model_performance)
    
