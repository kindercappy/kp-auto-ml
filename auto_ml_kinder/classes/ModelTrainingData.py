import pandas as pd
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from itertools import combinations
from auto_ml_kinder import model_feature_selector as mfs
from auto_ml_kinder.classes.Functions import Functions 
from auto_ml_kinder.classes.ScalerType import ScalerType
import numpy as np


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
    Y_train_neural_network:list = []
    Y_val_neural_network:list = []

    X_test:list = []
    Y_test_neural_network:list = []

    Normalizer = None
    Polynomializer = None
    PCAlizer = None
    Data_transformer_pipe:Pipeline = None

    Use_PCA = False
    Use_Polynomials = False
    Use_Feature_Selection = False
    create_clustering_feature_and_no_of_clusters = None
    clustering_model = None
    Selected_Features = []
    is_classification = False
    def __init__(self
                 ,df:pd.DataFrame
                 ,scaler_type:ScalerType
                 ,pca_variance = .95,
                 use_pca = False,
                 use_polynomials = False,
                 use_feature_selection = False,
                 create_clustering_feature_and_no_of_clusters = (False,3),
                 is_classification = False
                 ) -> None:
        
        func = Functions()
        self.Use_PCA = use_pca
        self.Use_Polynomials = use_polynomials
        self.Use_Feature_Selection = use_feature_selection
        self.create_clustering_feature_and_no_of_clusters = create_clustering_feature_and_no_of_clusters
        self.Selected_Features = df.columns
        self.is_classification = is_classification
        if(self.Use_Feature_Selection):
            tempX,tempY = func.get_x_y(df=df)
            columns_after_feature_selection = mfs.Run_Features_Selection(tempX,tempY)
            columns_after_feature_selection.append(tempY.name)# we need to add target column back to dataframe
            printable_names = ', '.join(columns_after_feature_selection)
            print(f'Total columns :{len(df.columns)}')
            print(f'Selected number of columns: {len(columns_after_feature_selection)}')
            print(f'Columns to use: {printable_names}')
            df = df[columns_after_feature_selection]
            self.Selected_Features = columns_after_feature_selection
            
        self.X,self.Y = func.get_x_y(df=df)
        self.X_original = self.X.copy(deep=True)
        self.num_features = self.X_original.shape[1]
        self.X_train_df,self.X_val_df,self.X_test_df = func.custom_train_val_test_df_split(df)
        self.X_train,self.Y_train = func.get_x_y(self.X_train_df)
        self.X_val,self.Y_val = func.get_x_y(self.X_val_df)
        self.X_test,self.Y_test = func.get_x_y(self.X_test_df)

        self.Y_train_neural_network = self.Y_train.copy()
        self.Y_test_neural_network = self.Y_test.copy()
        self.Y_val_neural_network = self.Y_val.copy()
        
        self.X_train_df,self.X_test_df,self.X_val_df = func.get_X(self.X_train_df),func.get_X(self.X_test_df),func.get_X(self.X_val_df)

        self.Normalizer = func.get_scaler(scaler_type)
        self.Polynomializer = PolynomialFeatures(degree=2)
        self.PCAlizer = PCA(n_components=pca_variance)

        total_columns = len(self.X_original.columns)
        pipeline = Pipeline([
            ('scaler', self.Normalizer)
        ])
        if(self.Use_Polynomials):
            pipeline.steps.append(('poly_features', self.Polynomializer))
        if(self.Use_PCA):
            pipeline.steps.append(('pca', self.PCAlizer))
            total_columns = pca_variance
        
        
        if(create_clustering_feature_and_no_of_clusters[0]):
            model = func.find_best_clustering_algorithm(self.X,num_clusters=create_clustering_feature_and_no_of_clusters[1])
            self.X,self.X_train,self.X_test,self.X_val,self.X_original,self.X_train_df,self.X_test_df,self.X_val_df = func.perform_clustering_on_first(
                self.X
                ,self.X_train
                ,self.X_test
                ,self.X_val
                ,self.X_original
                ,self.X_train_df
                ,self.X_test_df
                ,self.X_val_df
                ,num_clusters=create_clustering_feature_and_no_of_clusters[1]
                ,model=model)
            self.clustering_model = model
            self.Selected_Features = self.X_train_df.columns
            
        self.Data_transformer_pipe = pipeline
        
        if len(self.Data_transformer_pipe.steps) > 1:
            self.Data_transformer_pipe.fit(self.X_original)
            self.X = self.Data_transformer_pipe.transform(self.X)
            self.X_train = self.Data_transformer_pipe.transform(self.X_train)
            self.X_test = self.Data_transformer_pipe.transform(self.X_test)
            self.X_val = self.Data_transformer_pipe.transform(self.X_val)

            # self.X,self.X_train,self.X_test,self.X_val,self.X_original = perform_clustering_on_first(self.X,self.X_train,self.X_test,self.X_val,self.X_original)

        try:
            total_columns = len(self.X[0])
        except:
            total_columns = len(self.X.columns)

        print(f'Total columns being used after all data transformations: {total_columns}')

        if(self.is_classification):
            import keras as k
            num_classes = len(np.unique(self.Y))
            
            if num_classes > 2:
                print(f'Since num of classes is {num_classes} transforming Y_(test/train/val)_neural_network variables to categorical.')
                self.Y_test_neural_network = k.utils.to_categorical(self.Y_test)
                self.Y_train_neural_network = k.utils.to_categorical(self.Y_train)
                self.Y_val_neural_network = k.utils.to_categorical(self.Y_val)

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
                yield X_train_permuted, X_val_permuted, X_test_permuted,selected_columns

    def only_predict(self,X, clusterer):
        labels = clusterer.predict(X) if hasattr(clusterer, 'predict') else clusterer.fit_predict(X)
        return labels
