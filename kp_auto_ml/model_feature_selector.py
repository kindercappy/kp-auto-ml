import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso


def feature_importance_end_model_builder(feature_importance_dataframe,columns):
    beautified_df = pd.DataFrame({'column_name':columns,'value':feature_importance_dataframe[0]})
    return beautified_df

def run_feature_selection(df):
    print('Executing run_feature_selection which runs RandomForest, DecisionTree and Lasso models to find the features used in the model training.')
    important_features = pd.concat([RandomForestFeaturesSelection(df),DecisionTreeClassifierFeaturesSelection(df),LassoFeaturesSelection(df)])
    # print(important_features)
    important_features_non_duplicated = important_features[important_features.column_name.duplicated() == False]
    signal_df_processed_important_features = df.iloc[:,important_features_non_duplicated.column_name.to_numpy()]
    signal_df_processed_important_features[df.columns[len(df.columns)-1]] = df.iloc[:,-1]
    return signal_df_processed_important_features

def RandomForestFeaturesSelection(X:pd.DataFrame,Y:pd.Series):
    print('Executing RandomForestFeaturesSelection function to get the best features that are used in training the model.')
    model = RandomForestClassifier(random_state=1,criterion = 'entropy', max_depth=3)
    model.fit(X, Y)
    feature_importance = model.feature_importances_
    feature_importance_dataframe = pd.DataFrame(feature_importance)
    beautified_df = feature_importance_end_model_builder(feature_importance_dataframe,X.columns)
    comparer_get_more_than = beautified_df['value'].mean()
    temp2 = beautified_df[beautified_df['value'] > comparer_get_more_than].sort_values(by='value',ascending=False).reset_index(drop='index')
    return temp2.column_name.values

def DecisionTreeClassifierFeaturesSelection(X:pd.DataFrame,Y:pd.Series):
    print('Executing DecisionTreeClassifierFeaturesSelection function to get the best features that are used in training the model.')
    model = DecisionTreeClassifier(random_state=0, max_depth=3)
    model.fit(X, Y)
    feature_importance = model.feature_importances_
    feature_importance_dataframe = pd.DataFrame(feature_importance)
    beautified_df = feature_importance_end_model_builder(feature_importance_dataframe,X.columns)
    comparer_get_more_than = beautified_df['value'].mean()
    temp2 = beautified_df[beautified_df['value'] > comparer_get_more_than].sort_values(by='value',ascending=False).reset_index(drop='index')
    return temp2.column_name.values

def LassoFeaturesSelection(X:pd.DataFrame,Y:pd.Series):
    print('Executing LassoFeaturesSelection function to get the best features that are used in training the model.')
    model = Lasso(alpha=0.01)
    model.fit(X, Y)
    feature_importance = model.coef_
    feature_importance_dataframe = pd.DataFrame(feature_importance)
    beautified_df = feature_importance_end_model_builder(feature_importance_dataframe,X.columns)
    temp2 = beautified_df[beautified_df['value'] > 0].sort_values(by='value',ascending=False).reset_index(drop='index')
    return temp2.column_name.values


def Run_Features_Selection(X:pd.DataFrame,Y:pd.Series):
    array1 = RandomForestFeaturesSelection(X,Y)
    array2 = DecisionTreeClassifierFeaturesSelection(X,Y)
    array3 = LassoFeaturesSelection(X,Y)
    # Initialize an empty list to hold the combined values
    combined_array = []

    # Extend the combined_array with each array
    combined_array.extend(array1)
    combined_array.extend(array2)
    combined_array.extend(array3)

    # Get distinct values using set()
    distinct_values = set(combined_array)

    # Convert the set back to a list if needed
    distinct_values_list = list(distinct_values)
    return distinct_values_list