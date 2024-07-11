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
from auto_ml_kinder import model_feature_selector as mfs


def get_x_y(df:pd.DataFrame):
    X = df.iloc[:,0:-1]
    Y = df.iloc[:,-1]
    return X,Y

def get_X(df:pd.DataFrame):
    return df.iloc[:,0:-1]

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
        self.Use_PCA = use_pca
        self.Use_Polynomials = use_polynomials
        self.Use_Feature_Selection = use_feature_selection
        self.create_clustering_feature_and_no_of_clusters = create_clustering_feature_and_no_of_clusters
        self.Selected_Features = df.columns
        self.is_classification = is_classification
        if(self.Use_Feature_Selection):
            tempX,tempY = get_x_y(df=df)
            columns_after_feature_selection = mfs.Run_Features_Selection(tempX,tempY)
            columns_after_feature_selection.append(tempY.name)# we need to add target column back to dataframe
            printable_names = ', '.join(columns_after_feature_selection)
            print(f'Total columns :{len(df.columns)}')
            print(f'Selected number of columns: {len(columns_after_feature_selection)}')
            print(f'Columns to use: {printable_names}')
            df = df[columns_after_feature_selection]
            self.Selected_Features = columns_after_feature_selection
            
        self.X,self.Y = get_x_y(df=df)
        self.X_original = self.X.copy(deep=True)
        self.num_features = self.X_original.shape[1]
        self.X_train_df,self.X_val_df,self.X_test_df = custom_train_val_test_df_split(df)
        self.X_train,self.Y_train = get_x_y(self.X_train_df)
        self.X_val,self.Y_val = get_x_y(self.X_val_df)
        self.X_test,self.Y_test = get_x_y(self.X_test_df)

        self.Y_train_neural_network = self.Y_train.copy()
        self.Y_test_neural_network = self.Y_test.copy()
        self.Y_val_neural_network = self.Y_val.copy()
        
        self.X_train_df,self.X_test_df,self.X_val_df = get_X(self.X_train_df),get_X(self.X_test_df),get_X(self.X_val_df)

        self.Normalizer = get_scaler(scaler_type)
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
            model = find_best_clustering_algorithm(self.X,num_clusters=create_clustering_feature_and_no_of_clusters[1])
            self.X,self.X_train,self.X_test,self.X_val,self.X_original,self.X_train_df,self.X_test_df,self.X_val_df = perform_clustering_on_first(
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
                self.Y_test_neural_networkst = k.utils.to_categorical(self.Y_test)
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


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def perform_clustering_on_first(*X_data, num_clusters=10,model):
    if len(X_data) == 0:
        raise ValueError("At least one dataset must be provided.")
    
    result_data = []
    # return X_data
    scores = list()
    for X in X_data:
        labels = fit_and_predict(X,model)
        if isinstance(X, pd.DataFrame):
            X_with_clusters = X.copy()
            X_with_clusters[f'cluster_{num_clusters}'] = labels
            result_data.append(X_with_clusters)
        elif isinstance(X, np.ndarray):
            X_with_clusters = np.hstack((X, labels.reshape(-1, 1)))
            result_data.append(X_with_clusters)
        elif isinstance(X, list):
            X_with_clusters = [row + [label] for row, label in zip(X, labels)]
            result_data.append(X_with_clusters)

        from sklearn.metrics import silhouette_score
        scores.append(silhouette_score(X, labels))

    mean_score = sum(scores) / len(scores)
    print(f'Mean score of all clusters is: {mean_score}')
    return tuple(result_data)


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def find_best_clustering_algorithm(X, num_clusters=10):
    algorithms = {
        'kmeans': tune_kmeans(X,num_clusters),
        'agglomerative': tune_agglomerative(X,num_clusters),
        # 'spectral': tune_spectral(X),
        'mean_shift': tune_mean_shift(X),
        'dbscan': tune_dbscan(X),
        'birch': tune_birch(X),
        'affinity_propagation': tune_affinity_propagation(X),
        'mini_batch_kmeans': tune_mini_batch_kmeans(X,num_clusters)
    }

    best_score = -1
    best_algo_name = None
    best_algo = None
    
    for algo_name, clusterer in algorithms.items():
        labels = fit_and_predict(X, clusterer)

        num_labels = len(set(labels))
        if num_labels < 2 or num_labels >= len(X):
            print(f"Algorithm {algo_name} produced an invalid number of clusters: {num_labels}")
            continue

        score = silhouette_score(X, labels)
        print(f'Mean score for {algo_name} is: {score}')
        
        if score > best_score:
            best_score = score
            best_algo_name = algo_name
            best_algo = clusterer

    print(f'Best algorithm: {best_algo_name} with a score of: {best_score}')
    return algorithms[best_algo_name]

def fit_and_predict(X, clusterer):
    if hasattr(clusterer, 'predict'):
        clusterer.fit(X)
    else:
        clusterer.fit_predict(X)
    labels = clusterer.predict(X) if hasattr(clusterer, 'predict') else clusterer.fit_predict(X)
    return labels



import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score, homogeneity_score, v_measure_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, Birch, AffinityPropagation, MiniBatchKMeans

def tune_kmeans(X, num_clusters=10):
    print("Grid tuning in progress for KMeans...")
    kmeans_params = {'n_clusters': [num_clusters], 'random_state': [42]}
    kmeans = KMeans()
    kmeans_gs = GridSearchCV(kmeans, kmeans_params, scoring=adjusted_rand_score)
    kmeans_gs.fit(X)
    return kmeans_gs.best_estimator_

def tune_agglomerative(X, num_clusters=10):
    print("Grid tuning in progress for Agglomerative Clustering...")
    agglomerative_params = {'n_clusters': [num_clusters], 'linkage': ['ward', 'complete', 'average', 'single']}
    agglomerative = AgglomerativeClustering()
    agglomerative_gs = GridSearchCV(agglomerative, agglomerative_params, scoring=adjusted_rand_score)
    agglomerative_gs.fit(X)
    return agglomerative_gs.best_estimator_

def tune_spectral(X, num_clusters_range=(3, 5)):
    print("Grid tuning in progress for Spectral Clustering...")
    spectral_params = {'n_clusters': list(range(*num_clusters_range))}
    spectral = SpectralClustering()
    spectral_gs = GridSearchCV(spectral, spectral_params, scoring=adjusted_rand_score)
    spectral_gs.fit(X)
    return spectral_gs.best_estimator_

def tune_mean_shift(X):
    # MeanShift does not require tuning hyperparameters
    print("Mean Shift clustering does not require hyperparameter tuning.")
    return MeanShift()

def tune_dbscan(X):
    # DBSCAN does not require tuning hyperparameters
    print("DBSCAN clustering does not require hyperparameter tuning.")
    return DBSCAN()

def tune_birch(X):
    # Birch does not require tuning hyperparameters
    print("Birch clustering does not require hyperparameter tuning.")
    return Birch()

def tune_affinity_propagation(X):
    # AffinityPropagation does not require tuning hyperparameters
    print("Affinity Propagation clustering does not require hyperparameter tuning.")
    return AffinityPropagation()

def tune_mini_batch_kmeans(X,num_clusters=3):
    print("Grid tuning in progress for Mini-Batch KMeans...")
    mini_batch_kmeans_params = {'n_clusters': [num_clusters], 'random_state': [42]}
    mini_batch_kmeans = MiniBatchKMeans()
    mini_batch_kmeans_gs = GridSearchCV(mini_batch_kmeans, mini_batch_kmeans_params, scoring=adjusted_rand_score)
    mini_batch_kmeans_gs.fit(X)
    return mini_batch_kmeans_gs.best_estimator_