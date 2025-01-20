import pandas as pd
from auto_ml_kinder.classes.ModelAndParamRegression import ModelAndParamRegression
from auto_ml_kinder.classes.ModelPower import ModelPower
from auto_ml_kinder.classes.ModelAndParamClassifiction import ModelAndParamClassifiction
from auto_ml_kinder.classes.ModelPerformance import ModelPerformance
from auto_ml_kinder.classes.ScalerType import ScalerType
from auto_ml_kinder.model_list_helper import get_parameters_decision_tree_classifier,get_parameters_decision_tree_fit_reg,get_parameters_elasticnet_fit,get_parameters_gradient_boosting_classifier,get_parameters_gradient_boosting_fit_reg,get_parameters_knn_classifier,get_parameters_knn_reg,get_parameters_lasso_fit,get_parameters_linear_reg,get_parameters_logistic_reg,get_parameters_random_forest_classifier,get_parameters_random_forest_fit_reg,get_parameters_ridge_classifier,get_parameters_ridge_fit,get_parameters_svc_fit,get_parameters_svr_fit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score, homogeneity_score, v_measure_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, Birch, AffinityPropagation, MiniBatchKMeans
from sklearn.metrics import silhouette_score



class Functions():
    def get_model_and_param_regression(self,power:ModelPower,model_and_param:ModelAndParamRegression):
        if not isinstance(model_and_param, ModelAndParamRegression):
            raise ValueError("Invalid model_and_param type. Please provide a valid ModelAndParam enum value.")

        model = None
        param = None
        if model_and_param == ModelAndParamRegression.Linear_Regression:
            param,model = get_parameters_linear_reg()
        elif model_and_param == ModelAndParamRegression.Ridge_Regression:
            param,model = get_parameters_ridge_fit(power=power)
        elif model_and_param == ModelAndParamRegression.Lasso_Regression:
            param,model = get_parameters_lasso_fit(power=power)
        elif model_and_param == ModelAndParamRegression.ElasticNet_Regression:
            param,model = get_parameters_elasticnet_fit(power=power)
        elif model_and_param == ModelAndParamRegression.SVR_Regression:
            param,model = get_parameters_svr_fit(power=power)
        elif model_and_param == ModelAndParamRegression.DecisionTree_Regressor:
            param,model = get_parameters_decision_tree_fit_reg(power=power)
        elif model_and_param == ModelAndParamRegression.RandomForest_Regressor:
            param,model = get_parameters_random_forest_fit_reg(power=power)
        elif model_and_param == ModelAndParamRegression.GradientBoosting_Regressor:
            param,model = get_parameters_gradient_boosting_fit_reg(power=power)
        elif model_and_param == ModelAndParamRegression.KNeighbors_Regressor:
            param,model = get_parameters_knn_reg(power=power)

        return model,param

    def get_model_and_param_classification(self,power:ModelPower,model_and_param:ModelAndParamClassifiction):
        if not isinstance(model_and_param, ModelAndParamClassifiction):
            raise ValueError("Invalid model_and_param type. Please provide a valid ModelAndParam enum value.")

        model = None
        param = None
        if model_and_param == ModelAndParamClassifiction.Logistic_Regression:
            param,model = get_parameters_logistic_reg()
        elif model_and_param == ModelAndParamClassifiction.Ridge_Classifiction:
            param,model = get_parameters_ridge_classifier(power=power)
        elif model_and_param == ModelAndParamClassifiction.SVC_Classification:
            param,model = get_parameters_svc_fit(power=power)
        elif model_and_param == ModelAndParamClassifiction.DecisionTree_Classifiction:
            param,model = get_parameters_decision_tree_classifier(power=power)
        elif model_and_param == ModelAndParamClassifiction.RandomForest_Classifiction:
            param,model = get_parameters_random_forest_classifier(power=power)
        elif model_and_param == ModelAndParamClassifiction.GradientBoosting_Classifiction:
            param,model = get_parameters_gradient_boosting_classifier(power=power)
        elif model_and_param == ModelAndParamClassifiction.KNeighbors_Classifiction:
            param,model = get_parameters_knn_classifier(power=power)

        return model,param

    def train_test_random_search_regression(self,model, param_distributions, X_train, y_train, X_test, y_test, scoring='r2', cv=5):
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

    def train_test_random_search_classification(self,model, param_distributions, X_train, y_train, X_test, y_test, scoring='accuracy', cv=5):
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

        
    def insert_object_columns(self,df, obj:ModelPerformance):
        if df is None:
            df = pd.DataFrame(columns=obj.to_dict())
        
        row_data = obj.to_dict()
        
        df_new = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        
        return df_new

    
    def extract_last_digit_from_list(self,select_columns, match_string):
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
    

    def get_x_y(self,df:pd.DataFrame):
        X = df.iloc[:,0:-1]
        Y = df.iloc[:,-1]
        return X,Y

    def get_X(self,df:pd.DataFrame):
        return df.iloc[:,0:-1]

    def custom_train_val_test_df_split(self,df):
        X,Y = self.get_x_y(df)
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=1)
        X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=.3,random_state=1)
        train_df = pd.concat([X_train,Y_train], axis=1)
        val_df = pd.concat([X_val,Y_val], axis=1)
        test_df = pd.concat([X_test,Y_test],axis=1)
        return train_df,val_df,test_df



    def get_scaler(self,scaler_type):
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
    
    def find_best_clustering_algorithm(self,X, num_clusters=10):
        algorithms = {
            'kmeans': self.tune_kmeans(X,num_clusters),
            'agglomerative': self.tune_agglomerative(X,num_clusters),
            # 'spectral': tune_spectral(X),
            'mean_shift': self.tune_mean_shift(X),
            'dbscan': self.tune_dbscan(X),
            'birch': self.tune_birch(X),
            'affinity_propagation': self.tune_affinity_propagation(X),
            'mini_batch_kmeans': self.tune_mini_batch_kmeans(X,num_clusters)
        }
    def tune_kmeans(self,X, num_clusters=10):
        print("Grid tuning in progress for KMeans...")
        kmeans_params = {'n_clusters': [num_clusters], 'random_state': [42]}
        kmeans = KMeans()
        kmeans_gs = GridSearchCV(kmeans, kmeans_params, scoring=adjusted_rand_score)
        kmeans_gs.fit(X)
        return kmeans_gs.best_estimator_

    def tune_agglomerative(self,X, num_clusters=10):
        print("Grid tuning in progress for Agglomerative Clustering...")
        agglomerative_params = {'n_clusters': [num_clusters], 'linkage': ['ward', 'complete', 'average', 'single']}
        agglomerative = AgglomerativeClustering()
        agglomerative_gs = GridSearchCV(agglomerative, agglomerative_params, scoring=adjusted_rand_score)
        agglomerative_gs.fit(X)
        return agglomerative_gs.best_estimator_

    def tune_spectral(self,X, num_clusters_range=(3, 5)):
        print("Grid tuning in progress for Spectral Clustering...")
        spectral_params = {'n_clusters': list(range(*num_clusters_range))}
        spectral = SpectralClustering()
        spectral_gs = GridSearchCV(spectral, spectral_params, scoring=adjusted_rand_score)
        spectral_gs.fit(X)
        return spectral_gs.best_estimator_

    def tune_mean_shift(self,X):
        # MeanShift does not require tuning hyperparameters
        print("Mean Shift clustering does not require hyperparameter tuning.")
        return MeanShift()

    def tune_dbscan(self,X):
        # DBSCAN does not require tuning hyperparameters
        print("DBSCAN clustering does not require hyperparameter tuning.")
        return DBSCAN()

    def tune_birch(self,X):
        # Birch does not require tuning hyperparameters
        print("Birch clustering does not require hyperparameter tuning.")
        return Birch()

    def tune_affinity_propagation(self,X):
        # AffinityPropagation does not require tuning hyperparameters
        print("Affinity Propagation clustering does not require hyperparameter tuning.")
        return AffinityPropagation()

    def tune_mini_batch_kmeans(self,X,num_clusters=3):
        print("Grid tuning in progress for Mini-Batch KMeans...")
        mini_batch_kmeans_params = {'n_clusters': [num_clusters], 'random_state': [42]}
        mini_batch_kmeans = MiniBatchKMeans()
        mini_batch_kmeans_gs = GridSearchCV(mini_batch_kmeans, mini_batch_kmeans_params, scoring=adjusted_rand_score)
        mini_batch_kmeans_gs.fit(X)
        return mini_batch_kmeans_gs.best_estimator_

    def find_best_clustering_algorithm(self,X, num_clusters=10):
        algorithms = {
            'kmeans': self.tune_kmeans(X,num_clusters),
            'agglomerative': self.tune_agglomerative(X,num_clusters),
            # 'spectral': tune_spectral(X),
            'mean_shift': self.tune_mean_shift(X),
            'dbscan':self. tune_dbscan(X),
            'birch': self.tune_birch(X),
            'affinity_propagation': self.tune_affinity_propagation(X),
            'mini_batch_kmeans': self.tune_mini_batch_kmeans(X,num_clusters)
        }

        best_score = -1
        best_algo_name = None
        best_algo = None
        
        for algo_name, clusterer in algorithms.items():
            labels = self.fit_and_predict(X, clusterer)

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
    
    def fit_and_predict(self,X, clusterer):
        if hasattr(clusterer, 'predict'):
            clusterer.fit(X)
        else:
            clusterer.fit_predict(X)
        labels = clusterer.predict(X) if hasattr(clusterer, 'predict') else clusterer.fit_predict(X)
        return labels

    def perform_clustering_on_first(self,*X_data, num_clusters=10,model):
        if len(X_data) == 0:
            raise ValueError("At least one dataset must be provided.")
        
        result_data = []
        # return X_data
        scores = list()
        for X in X_data:
            labels = self.fit_and_predict(X,model)
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
