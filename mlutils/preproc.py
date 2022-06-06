import pandas as pd
import os
from sklearn.impute import SimpleImputer

def check_duplicates(df: pd.DataFrame) -> bool:
    return df.duplicated().sum() > 0

def check_na(df: pd.DataFrame):
    """
    Renvoi une série contenant le pourcentage de NA par colonnes
    """
    return df.isna().sum().sort_values(ascending=False) / df.shape[0]


def simple_imputer_by_columns(
    train_df: pd.DataFrame,
    columns_def: dict,
) -> pd.DataFrame:
    """
    Renvoi un DataFrame où les valeurs manquantes sont imputés selon les
    paramètres passés en argument.

    ## Parameters
    - train_df: DataFrame
    - columns_def: dictionnaire où chaque clef est un nom de colonne du DataFrame et où
    les clefs / valeurs sont les paramètres pour l'imputer

    ## Return
    Renvoi un DataFrame où les *na* des colonnes passées en argument sont remplis
    """

    for column_name, imputer_args in columns_def.items():
        imp_mean = SimpleImputer(**imputer_args)
        train_df[column_name] = imp_mean.fit_transform(train_df[[column_name]])

    return train_df




if __name__ == '__main__':

    import numpy as np


    a = pd.DataFrame({
        'col1': [1, np.nan, 2, 3],
        'col2': ['a', 'b', '-', 'c']
    })

    a = simple_imputer_by_columns(
        a,
        columns_def = {
            'col1' : {
                'missing_values' : np.nan,
                'strategy' : 'mean',
            },
            'col2': {
                'missing_values' : '-',
                'fill_value': "aaaaaaaaa",
                'strategy': 'constant'
            }
        }
    )
