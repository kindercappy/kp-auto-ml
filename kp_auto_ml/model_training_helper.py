from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from kp_auto_ml import model_training_data_prep as dp

from enum import Enum


class ModelPower(Enum):
    LITE = 'li'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


def get_parameters_linear_reg():
    linear_reg_hyper_params = {
        'fit_intercept': [True, False]
    }
    print(f'LinearRegression params: {linear_reg_hyper_params}')
    return linear_reg_hyper_params, LinearRegression()

def get_parameters_ridge_fit(power:ModelPower):
    alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    if power == ModelPower.LOW:
        alpha_options = [0.1, 0.5, 1.0]
    elif power == ModelPower.MEDIUM:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0]
    elif power == ModelPower.LITE:
        alpha_options = [0.1, 0.5]
    elif power == ModelPower.HIGH:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    ridge_hyper_params = {
        'alpha': alpha_options,
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    print(f'Ridge params: {ridge_hyper_params}')
    return ridge_hyper_params, Ridge()




class ModelAndParam(Enum):
    Linear_Regression= LinearRegression
    Ridge_Regression=Ridge

def get_model_and_param(power:ModelPower,model_and_param:ModelAndParam):
    if not isinstance(model_and_param, ModelAndParam):
        raise ValueError("Invalid model_and_param type. Please provide a valid ModelAndParam enum value.")

    model = None
    param = None
    if model_and_param == ModelAndParam.Linear_Regression:
        model,param = get_parameters_linear_reg()
    elif model_and_param == ModelAndParam.Ridge_Regression:
        model,param = get_parameters_ridge_fit(power=power)

    # if model & param is None:
    #     raise ValueError("Invalid model_and_param type.")

    return model,param



def train_test_random_search_regression(model, param_distributions, X_train, y_train, X_test, y_test, n_iter=10, scoring='r2', cv=5):
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, scoring=scoring, cv=cv, random_state=42)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    
    best_model.fit(X_train, y_train)
    
    test_score = best_model.score(X_test, y_test)
    
    return best_params, best_model, test_score