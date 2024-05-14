import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

class LabelEncodingDict():
    column_value_to_replace = None
    column_value_to_replace_with = None
    def __init__(self,column_value_to_replace, column_value_to_replace_with) -> None:
        self.column_value_to_replace = column_value_to_replace
        self.column_value_to_replace_with = column_value_to_replace_with

class PreLabelEncoderConfig():
    column_name = None
    label_encoding:list[LabelEncodingDict] = None
    def __init__(self, column_name,label_encoding):
        self.column_name = column_name
        self.label_encoding = label_encoding
    def convert_to_dict(self):
        return dict((obj.column_value_to_replace, obj.column_value_to_replace_with) for obj in self.label_encoding)
        

class PreNumericColDataChangeConfig():
    column_name = None
    data_type = None
    def __init__(self,col_name,data_type) -> None:
        self.column_name = col_name
        self.data_type = data_type

class PreProcessingConfig():
    encoding_dummies:list[str]
    label_encode:list[PreLabelEncoderConfig]
    numeric_cols_data_changer:list[PreNumericColDataChangeConfig]
    target_column:str
    exclude_columns:list[str]


    def __init__(
            self
            ,encoding_dummies:list[str]
            ,label_encode:list[PreLabelEncoderConfig]
            ,numeric_cols_data_changer:list[PreNumericColDataChangeConfig]
            ,exclude_columns
            ,target_column = None
            ):
        self.encoding_dummies = encoding_dummies
        self.label_encode = label_encode
        self.numeric_cols_data_changer = numeric_cols_data_changer
        self.target_column = target_column
        self.exclude_columns = exclude_columns

def fillna(df):
    print('Filling NA values with median and mode for numeria and non-numeric columns.')
    df_internal = pd.DataFrame(df)
    for col in df_internal.columns:
        if(is_numeric_dtype(df_internal[col])):
            df_internal[col] = df_internal[col].fillna(df_internal[col].median())
        else:
            df_internal[col] = df_internal[col].fillna(df_internal[col].value_counts().idxmax())
    return df_internal


def remove_duplicates(df, subset=None, keep='first'):
    print('Removing duplicates.')
    df_unique = df.drop_duplicates(subset=subset, keep=keep)
    return df_unique



def null_unsuable_values_cleaner(frame, dirty_symbols_with_replacer_value=[{' ': np.nan}, {'?': np.nan}]):
    print('Removing unusable values like " " and "?"')
    df = pd.DataFrame(frame)
    for x in dirty_symbols_with_replacer_value:
        df = df.replace(x)
    return df


def numeric_columns_data_changer(df,cols_and_datatype:list[PreNumericColDataChangeConfig]):
    print('Changing datatype of numeric columns.')
    df_internal = pd.DataFrame(df)
    for x in cols_and_datatype:
        df_internal[x.column_name] = df_internal[x.column_name].astype(x.data_type)
    return df_internal

def label_encoding(df,columns_with_label_encoding:list[PreLabelEncoderConfig]):
    print('Performing label encoding on columns provided.')
    df_internal = pd.DataFrame(df)
    for x in columns_with_label_encoding:
        df_internal[x.column_name] =  df_internal[x.column_name].replace(x.convert_to_dict())
    return df_internal

def encoding_dummies(df,columns_to_dummise):
    print('Encoding using dummies.')
    df_internal = pd.DataFrame(df)
    df_internal = pd.get_dummies(df_internal, columns=columns_to_dummise)
    return df_internal

def handle_outliers_iqr(df,target_col, iqr_multiplier=1.5):
    print(f'Performing IQR based outliers cleaning on target. {target_col}')
    df_outliers_removed = df.copy()
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)

    df_outliers_removed = df_outliers_removed[(df_outliers_removed[target_col] >= lower_bound) & (df_outliers_removed[target_col] <= upper_bound)]

    return df_outliers_removed

def process(df:pd.DataFrame,model_config:PreProcessingConfig):

    df_preprocessing = dropping_cols_rows(df)
    df_preprocessing = df_preprocessing.drop(columns=model_config.exclude_columns)
    df_preprocessing = null_unsuable_values_cleaner(df_preprocessing)
    df_preprocessing = fillna(df_preprocessing)
    df_preprocessing = handle_outliers_iqr(df_preprocessing,model_config.target_column)
    df_preprocessing = numeric_columns_data_changer(df_preprocessing, model_config.numeric_cols_data_changer)
    df_preprocessing = label_encoding(df_preprocessing,model_config.label_encode)
    df_preprocessing = encoding_dummies(df_preprocessing, model_config.encoding_dummies)
    df_preprocessing = df_preprocessing[[col for col in df_preprocessing.columns if col != model_config.target_column] + [model_config.target_column]]
    return df_preprocessing.reset_index(drop=True)

def process_test(df:pd.DataFrame,model_config:PreProcessingConfig):

    df_preprocessing = dropping_cols_rows(df)
    df_preprocessing = df_preprocessing.drop(columns=model_config.exclude_columns)
    df_preprocessing = null_unsuable_values_cleaner(df_preprocessing)
    df_preprocessing = fillna(df_preprocessing)
    # df_preprocessing = handle_outliers_iqr(df_preprocessing,model_config.target_column)
    df_preprocessing = numeric_columns_data_changer(df_preprocessing, model_config.numeric_cols_data_changer)
    df_preprocessing = label_encoding(df_preprocessing,model_config.label_encode)
    df_preprocessing = encoding_dummies(df_preprocessing, model_config.encoding_dummies)
    # df_preprocessing = df_preprocessing[[col for col in df_preprocessing.columns if col != model_config.target_column] + [model_config.target_column]]
    return df_preprocessing.reset_index(drop=True)

def dropping_cols_rows(df,threshold = .7):
    print(f'Dropping columns whose values are nullable greater than {threshold}')
    df_internal = pd.DataFrame(df)
    for col in df_internal.columns:
        df_internal = df_internal[df_internal.columns[df_internal.isnull().mean() < threshold]]

        df_internal = df_internal.loc[df_internal.isnull().mean(axis=1) < threshold]
    return df_internal




