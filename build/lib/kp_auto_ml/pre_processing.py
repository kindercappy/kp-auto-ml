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
            ,target_column
            ,exclude_columns
            ):
        self.encoding_dummies = encoding_dummies
        self.label_encode = label_encode
        self.numeric_cols_data_changer = numeric_cols_data_changer
        self.target_column = target_column
        self.exclude_columns = exclude_columns

def fillna(df):
    df_internal = pd.DataFrame(df)
    for col in df_internal.columns:
        if(is_numeric_dtype(df_internal[col])):
            df_internal[col] = df_internal[col].fillna(df_internal[col].median())
        else:
            df_internal[col] = df_internal[col].fillna(df_internal[col].value_counts().idxmax())
    return df_internal


def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - subset: Columns to consider for identifying duplicates. Default is None (use all columns).
    - keep: {'first', 'last', False}, default 'first'. Specifies which duplicates to keep.

    Returns:
    - df_unique: DataFrame with duplicate rows removed.
    """
    df_unique = df.drop_duplicates(subset=subset, keep=keep)
    return df_unique



def null_unsuable_values_cleaner(frame, dirty_symbols_with_replacer_value=[{' ': np.nan}, {'?': np.nan}]):
    df = pd.DataFrame(frame)
    for x in dirty_symbols_with_replacer_value:
        df = df.replace(x)
    return df


def numeric_columns_data_changer(df,cols_and_datatype:list[PreNumericColDataChangeConfig]):
    df_internal = pd.DataFrame(df)
    for x in cols_and_datatype:
        df_internal[x.column_name] = df_internal[x.column_name].astype(x.data_type)
    return df_internal

def label_encoding(df,columns_with_label_encoding:list[PreLabelEncoderConfig]):
    df_internal = pd.DataFrame(df)
    for x in columns_with_label_encoding:
        df_internal[x.column_name] =  df_internal[x.column_name].replace(x.convert_to_dict())
    return df_internal

def encoding_dummies(df,columns_to_dummise):
    df_internal = pd.DataFrame(df)
    df_internal = pd.get_dummies(df_internal, columns=columns_to_dummise)
    return df_internal


def process(df:pd.DataFrame,model_config:PreProcessingConfig):

    df_preprocessing = dropping_cols_rows(df,)
    df_preprocessing = null_unsuable_values_cleaner(df_preprocessing)
    df_preprocessing = numeric_columns_data_changer(df_preprocessing, model_config.numeric_cols_data_changer)
    df_preprocessing = label_encoding(df_preprocessing,model_config.label_encode)
    df_preprocessing = encoding_dummies(df_preprocessing, model_config.encoding_dummies)
    df_preprocessing = fillna(df_preprocessing)
    
    return df_preprocessing



def dropping_cols_rows(df,threshold = .7):
    df_internal = pd.DataFrame(df)
    for col in df_internal.columns:
        df_internal = df_internal[df_internal.columns[df_internal.isnull().mean() < threshold]]

        df_internal = df_internal.loc[df_internal.isnull().mean(axis=1) < threshold]
    return df_internal




