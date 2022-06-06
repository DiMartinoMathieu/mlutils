import os
import pandas as pd
import numpy as np
import mlutils.preproc as pproc

def test_duplicates():
    test_file = os.path.join(os.path.dirname(__file__), 'duplicated.csv')
    df = pd.read_csv(test_file)
    assert pproc.check_duplicates(df) == True

def test_na():
    test_file = os.path.join(os.path.dirname(__file__), 'duplicated.csv')
    df = pd.read_csv(test_file)
    na_on_product_id = pproc.check_na(df)['product_id']
    assert round(na_on_product_id, 2) == 0.17

def test_simple_imputer():
    a = pd.DataFrame({
        'col1': [1, np.nan, 2, 3],
        'col2': ['a', 'b', '-', 'c']
    })

    a = pproc.simple_imputer_by_columns(
        a,
        columns_def = {
            'col1' : {
                'missing_values' : np.nan,
                'strategy' : 'mean',
            },
            'col2': {
                'missing_values' : '-',
                'fill_value': "aaa",
                'strategy': 'constant'
            }
        }
    )

    assert a.equals(pd.DataFrame({
        'col1': [1.0, 2.0, 2.0, 3.0],
        'col2': ['a', 'b', 'aaa', 'c']
    }))
