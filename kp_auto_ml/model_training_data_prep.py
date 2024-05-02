from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import itertools
from itertools import combinations
import math


def get_x_y(df:pd.DataFrame):
    X = df.iloc[:,0:-1]
    Y = df.iloc[:,-1]
    return X,Y

def custom_train_val_test_df_split(df):
    X,Y = get_x_y(df)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=1)
    X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=.3,random_state=1)
    train_df = pd.concat([X_train,Y_train], axis=1)
    val_df = pd.concat([X_val,Y_val], axis=1)
    test_df = pd.concat([X_test,Y_test],axis=1)
    return train_df,val_df,test_df

class ScalerType(Enum):
    STANDARD_SCALER = StandardScaler
    MINMAX_SCALER = MinMaxScaler
    MAXABS_SCALER = MaxAbsScaler
    ROBUST_SCALER = RobustScaler
    QUANTILE_TRANSFORMER = QuantileTransformer

def get_scaler(scaler_type):
    if not isinstance(scaler_type, ScalerType):
        raise ValueError("Invalid scaler type. Please provide a valid ScalerType enum value.")

    scaler = None
    if scaler_type == ScalerType.STANDARD_SCALER:
        scaler = StandardScaler()
    elif scaler_type == ScalerType.MINMAX_SCALER:
        scaler = MinMaxScaler()
    elif scaler_type == ScalerType.MAXABS_SCALER:
        scaler = MaxAbsScaler()
    elif scaler_type == ScalerType.ROBUST_SCALER:
        scaler = RobustScaler()
    elif scaler_type == ScalerType.QUANTILE_TRANSFORMER:
        scaler = QuantileTransformer()

    if scaler is None:
        raise ValueError("Invalid scaler type.")

    return scaler

class ModelTrainingData():
    X_original:pd.DataFrame = None
    num_features = 0
    X:pd.DataFrame = None
    Y:pd.Series = None
    X_train_df:pd.DataFrame = None
    X_val_df:pd.DataFrame = None
    X_test_df:pd.DataFrame = None

    X_train:list = []
    X_val:list = []
    Y_train:list = []
    Y_val:list = []

    X_test:list = []
    Y_test:list = []

    Normalizer = None
    Polynomializer = None
    PCAlizer = None
    Data_transformer_pipe:Pipeline = None
    def __init__(self
                 ,df:pd.DataFrame
                 ,scaler_type:ScalerType
                 ,pca_variance = .95
                 ) -> None:
        self.X,self.Y = get_x_y(df=df)
        self.X_original = self.X.copy(deep=True)
        self.num_features = self.X_original.shape[1]
        self.X_train_df,self.X_val_df,self.X_test_df = custom_train_val_test_df_split(df)
        self.X_train,self.Y_train = get_x_y(self.X_train_df)
        self.X_val,self.Y_val = get_x_y(self.X_val_df)
        self.X_test,self.Y_test = get_x_y(self.X_test_df)

        self.Normalizer = get_scaler(scaler_type)
        self.Polynomializer = PolynomialFeatures(degree=2)
        self.PCAlizer = PCA(n_components=pca_variance)

        pipeline = Pipeline([
            ('scaler', self.Normalizer),  # StandardScaler for scaling
            # ('poly_features', self.Polynomializer),  # Example degree of 2, adjust as needed
            # ('pca', self.PCAlizer)  # Retain 95% of variance, adjust as needed
        ])
        self.Data_transformer_pipe = pipeline
        
        if len(self.Data_transformer_pipe.steps) > 1:
            self.Data_transformer_pipe.fit(self.X_original)
            self.X = self.Data_transformer_pipe.transform(self.X)
            self.X_train = self.Data_transformer_pipe.transform(self.X_train)
            self.X_test = self.Data_transformer_pipe.transform(self.X_test)
            self.X_val = self.Data_transformer_pipe.transform(self.X_val)
    
    def generate_permutations_train(self, min_columns):
        num_features = self.X_original.shape[1]
        total_permutations = sum(len(list(combinations(range(num_features), r))) for r in range(min_columns, num_features + 1))
        print("Total Permutations:", total_permutations)

        columns = self.X_original.columns
        for r in range(min_columns, num_features + 1):
            for combo in combinations(range(num_features), r):
                selected_columns = [columns[i] for i in list(combo)]
                if len(self.Data_transformer_pipe.steps) == 0:
                    X_train_permuted = self.X_train_df.loc[:, selected_columns].values
                    X_val_permuted = self.X_val_df.loc[:, selected_columns].values
                    X_test_permuted = self.X_test_df.loc[:, selected_columns].values
                    # yield X_train_permuted, X_val_permuted, X_test_permuted
                else:
                    X_train_permuted = self.Data_transformer_pipe.fit_transform(self.X_train_df.loc[:, selected_columns])
                    X_val_permuted = self.Data_transformer_pipe.transform(self.X_val_df.loc[:, selected_columns])
                    X_test_permuted = self.Data_transformer_pipe.transform(self.X_test_df.loc[:, selected_columns])
                yield X_train_permuted, X_val_permuted, X_test_permuted






