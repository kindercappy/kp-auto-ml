from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from auto_ml_kinder.classes.ModelPower import ModelPower

def get_parameters_linear_reg():
    linear_reg_hyper_params = {
        'fit_intercept': [True, False]
    }
    # print(f'LinearRegression params: {linear_reg_hyper_params}')
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
    # print(f'Ridge params: {ridge_hyper_params}')
    return ridge_hyper_params, Ridge()

def get_parameters_lasso_fit(power):
    alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    if power == ModelPower.LOW:
        alpha_options = [0.1, 0.5, 1.0]
    elif power == ModelPower.MEDIUM:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0]
    elif power == ModelPower.LITE:
        alpha_options = [0.1, 0.5]
    elif power == ModelPower.HIGH:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    lasso_hyper_params = {
        'alpha': alpha_options,
        'selection': ['cyclic', 'random']
    }
    # print(f'Lasso params: {lasso_hyper_params}')
    return lasso_hyper_params, Lasso()

def get_parameters_elasticnet_fit(power):
    alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    if power == ModelPower.LOW:
        alpha_options = [0.1, 0.5, 1.0]
    elif power == ModelPower.MEDIUM:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0]
    elif power == ModelPower.LITE:
        alpha_options = [0.1, 0.5]
    elif power == ModelPower.HIGH:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    l1_ratio_options = [0.1, 0.3, 0.5, 0.7, 0.9]
    elasticnet_hyper_params = {
        'alpha': alpha_options,
        'l1_ratio': l1_ratio_options,
        # 'normalize': [True, False],
        'selection': ['cyclic', 'random']
    }
    # print(f'ElasticNet params: {elasticnet_hyper_params}')
    return elasticnet_hyper_params, ElasticNet()

def get_parameters_svr_fit(power):
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    if power == ModelPower.LOW:
        C_options = [0.1, 1.0]
        epsilon_options = [0.1, 0.5]
    elif power == ModelPower.MEDIUM:
        C_options = [0.1, 1.0, 10.0]
        epsilon_options = [0.1, 0.5, 1.0]
    elif power == ModelPower.LITE:
        C_options = [0.1, 1.0]
        epsilon_options = [0.1]
    elif power == ModelPower.HIGH:
        C_options = [0.1, 1.0, 10.0, 100.0]
        epsilon_options = [0.1, 0.5, 1.0, 2.0]

    svr_hyper_params = {
        'kernel': kernel_options,
        'C': C_options,
        'epsilon': epsilon_options,
        'gamma': ['scale', 'auto']
    }
    # print(f'SVR params: {svr_hyper_params}')
    return svr_hyper_params, SVR()

def get_parameters_decision_tree_fit_reg(power):
    max_depth_options = [None, 5, 10, 20]
    min_samples_split_options = [2, 5, 10]
    if power == ModelPower.LOW:
        max_depth_options = [None, 5]
        min_samples_split_options = [2, 5]
    elif power == ModelPower.MEDIUM:
        max_depth_options = [None, 5, 10]
        min_samples_split_options = [2, 5, 10]
    elif power == ModelPower.LITE:
        max_depth_options = [None]
        min_samples_split_options = [2]
    elif power == ModelPower.HIGH:
        max_depth_options = [None, 5, 10, 20, 50]
        min_samples_split_options = [2, 5, 10, 20]

    decision_tree_hyper_params = {
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    # print(f'DecisionTree params: {decision_tree_hyper_params}')
    return decision_tree_hyper_params, DecisionTreeRegressor()

def get_parameters_random_forest_fit_reg(power):
    n_estimators_options = [50, 100, 200]
    max_depth_options = [None, 5, 10, 20]
    min_samples_split_options = [2, 5, 10]
    if power == ModelPower.LOW:
        n_estimators_options = [50, 100]
        max_depth_options = [None, 5]
        min_samples_split_options = [2, 5]
    elif power == ModelPower.MEDIUM:
        n_estimators_options = [50, 100, 200]
        max_depth_options = [None, 5, 10]
        min_samples_split_options = [2, 5, 10]
    elif power == ModelPower.LITE:
        n_estimators_options = [50]
        max_depth_options = [None]
        min_samples_split_options = [2]
    elif power == ModelPower.HIGH:
        n_estimators_options = [50, 100, 200, 500]
        max_depth_options = [None, 5, 10, 20]
        min_samples_split_options = [2, 5, 10, 20]

    random_forest_hyper_params = {
        'n_estimators': n_estimators_options,
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    # print(f'RandomForest params: {random_forest_hyper_params}')
    return random_forest_hyper_params, RandomForestRegressor()

def get_parameters_gradient_boosting_fit_reg(power):
    n_estimators_options = [50, 100, 200]
    learning_rate_options = [0.01, 0.1, 0.5]
    max_depth_options = [3, 5, 7]
    min_samples_split_options = [2, 5, 10]
    if power == ModelPower.LOW:
        n_estimators_options = [50, 100]
        learning_rate_options = [0.01, 0.1]
        max_depth_options = [3, 5]
        min_samples_split_options = [2, 5]
    elif power == ModelPower.MEDIUM:
        n_estimators_options = [50, 100, 200]
        learning_rate_options = [0.01, 0.1, 0.5]
        max_depth_options = [3, 5, 7]
        min_samples_split_options = [2, 5, 10]
    elif power == ModelPower.LITE:
        n_estimators_options = [50]
        learning_rate_options = [0.01]
        max_depth_options = [3]
        min_samples_split_options = [2]
    elif power == ModelPower.HIGH:
        n_estimators_options = [50, 100, 200, 500]
        learning_rate_options = [0.01, 0.1, 0.5]
        max_depth_options = [3, 5, 7, 10]
        min_samples_split_options = [2, 5, 10, 20]

    gradient_boosting_hyper_params = {
        'n_estimators': n_estimators_options,
        'learning_rate': learning_rate_options,
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    # print(f'GradientBoosting params: {gradient_boosting_hyper_params}')
    return gradient_boosting_hyper_params, GradientBoostingRegressor()

def get_parameters_knn_reg(power):
    n_neighbors_values = [3, 5, 7, 10, 15]
    weights_options = ['uniform', 'distance']
    
    if power == ModelPower.LOW:
        n_neighbors_values = [3, 5, 7]
    elif power == ModelPower.MEDIUM:
        n_neighbors_values = [3, 5, 7, 10]
    elif power == ModelPower.LITE:
        n_neighbors_values = [3, 5]
    elif power == ModelPower.HIGH:
        n_neighbors_values = [3, 5, 7, 10, 15]

    knn_reg_hyper_params = {
        'n_neighbors': n_neighbors_values,
        'weights': weights_options
    }
    # print(f'KNN params: {knn_reg_hyper_params}')
    return knn_reg_hyper_params, KNeighborsRegressor()


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from enum import Enum

def get_parameters_logistic_reg():
    logistic_reg_hyper_params = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg']
    }
    return logistic_reg_hyper_params, LogisticRegression()

def get_parameters_ridge_classifier(power: ModelPower):
    alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    if power == ModelPower.LOW:
        alpha_options = [0.1, 0.5, 1.0]
    elif power == ModelPower.MEDIUM:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0]
    elif power == ModelPower.LITE:
        alpha_options = [0.1, 0.5]
    elif power == ModelPower.HIGH:
        alpha_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    ridge_classifier_hyper_params = {
        'alpha': alpha_options,
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    return ridge_classifier_hyper_params, RidgeClassifier()

def get_parameters_svc_fit(power: ModelPower):
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    if power == ModelPower.LOW:
        C_options = [0.1, 1.0]
        gamma_options = ['scale']
    elif power == ModelPower.MEDIUM:
        C_options = [0.1, 1.0, 10.0]
        gamma_options = ['scale', 'auto']
    elif power == ModelPower.LITE:
        C_options = [0.1, 1.0]
        gamma_options = ['scale']
    elif power == ModelPower.HIGH:
        C_options = [0.1, 1.0, 10.0, 100.0]
        gamma_options = ['scale', 'auto']

    svc_hyper_params = {
        'kernel': kernel_options,
        'C': C_options,
        'gamma': gamma_options
    }
    return svc_hyper_params, SVC()

def get_parameters_decision_tree_classifier(power: ModelPower):
    max_depth_options = [None, 5, 10, 20]
    min_samples_split_options = [2, 5, 10]
    if power == ModelPower.LOW:
        max_depth_options = [None, 5]
        min_samples_split_options = [2, 5]
    elif power == ModelPower.MEDIUM:
        max_depth_options = [None, 5, 10]
        min_samples_split_options = [2, 5, 10]
    elif power == ModelPower.LITE:
        max_depth_options = [None]
        min_samples_split_options = [2]
    elif power == ModelPower.HIGH:
        max_depth_options = [None, 5, 10, 20, 50]
        min_samples_split_options = [2, 5, 10, 20]

    decision_tree_hyper_params = {
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    return decision_tree_hyper_params, DecisionTreeClassifier()

def get_parameters_random_forest_classifier(power: ModelPower):
    n_estimators_options = [50, 100, 200]
    max_depth_options = [None, 5, 10, 20]
    min_samples_split_options = [2, 5, 10]
    if power == ModelPower.LOW:
        n_estimators_options = [50, 100]
        max_depth_options = [None, 5]
        min_samples_split_options = [2, 5]
    elif power == ModelPower.MEDIUM:
        n_estimators_options = [50, 100, 200]
        max_depth_options = [None, 5, 10]
        min_samples_split_options = [2, 5, 10]
    elif power == ModelPower.LITE:
        n_estimators_options = [50]
        max_depth_options = [None]
        min_samples_split_options = [2]
    elif power == ModelPower.HIGH:
        n_estimators_options = [50, 100, 200, 500]
        max_depth_options = [None, 5, 10, 20]
        min_samples_split_options = [2, 5, 10, 20]

    random_forest_hyper_params = {
        'n_estimators': n_estimators_options,
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    return random_forest_hyper_params, RandomForestClassifier()

def get_parameters_gradient_boosting_classifier(power: ModelPower):
    n_estimators_options = [50, 100, 200]
    learning_rate_options = [0.01, 0.1, 0.5]
    max_depth_options = [3, 5, 7]
    min_samples_split_options = [2, 5, 10]
    if power == ModelPower.LOW:
        n_estimators_options = [50, 100]
        learning_rate_options = [0.01, 0.1]
        max_depth_options = [3, 5]
        min_samples_split_options = [2, 5]
    elif power == ModelPower.MEDIUM:
        n_estimators_options = [50, 100, 200]
        learning_rate_options = [0.01, 0.1, 0.5]
        max_depth_options = [3, 5, 7]
        min_samples_split_options = [2, 5, 10]
    elif power == ModelPower.LITE:
        n_estimators_options = [50]
        learning_rate_options = [0.01]
        max_depth_options = [3]
        min_samples_split_options = [2]
    elif power == ModelPower.HIGH:
        n_estimators_options = [50, 100, 200, 500]
        learning_rate_options = [0.01, 0.1, 0.5]
        max_depth_options = [3, 5, 7, 10]
        min_samples_split_options = [2, 5, 10, 20]

    gradient_boosting_hyper_params = {
        'n_estimators': n_estimators_options,
        'learning_rate': learning_rate_options,
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    return gradient_boosting_hyper_params, GradientBoostingClassifier()

def get_parameters_knn_classifier(power: ModelPower):
    n_neighbors_values = [3, 5, 7, 10, 15]
    weights_options = ['uniform', 'distance']
    
    if power == ModelPower.LOW:
        n_neighbors_values = [3, 5, 7]
    elif power == ModelPower.MEDIUM:
        n_neighbors_values = [3, 5, 7, 10]
    elif power == ModelPower.LITE:
        n_neighbors_values = [3, 5]
    elif power == ModelPower.HIGH:
        n_neighbors_values = [3, 5, 7, 10, 15, 20]
    
    knn_hyper_params = {
        'n_neighbors': n_neighbors_values,
        'weights': weights_options
    }
    
    return knn_hyper_params, KNeighborsClassifier()


