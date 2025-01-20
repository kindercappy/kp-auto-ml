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

class ModelAndParamRegression(Enum):
    Linear_Regression = LinearRegression
    Ridge_Regression = Ridge
    Lasso_Regression = Lasso
    ElasticNet_Regression = ElasticNet
    SVR_Regression = SVR
    DecisionTree_Regressor = DecisionTreeRegressor
    RandomForest_Regressor = RandomForestRegressor
    GradientBoosting_Regressor = GradientBoostingRegressor
    KNeighbors_Regressor = KNeighborsRegressor