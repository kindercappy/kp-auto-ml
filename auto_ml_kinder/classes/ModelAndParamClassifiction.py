from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from enum import Enum

class ModelAndParamClassifiction(Enum):
    Logistic_Regression = LogisticRegression
    Ridge_Classifiction = RidgeClassifier
    SVC_Classification = SVC
    DecisionTree_Classifiction = DecisionTreeClassifier
    RandomForest_Classifiction = RandomForestClassifier
    GradientBoosting_Classifiction = GradientBoostingClassifier
    KNeighbors_Classifiction = KNeighborsClassifier