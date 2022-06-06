# MLUTILS

## preproc
- check_duplicates: renvoi si des duplicates sont dans le dataframe
- check_na: renvoi la proportion de NA par colonnes
### simple_imputer_by_columns
Renvoi un DataFrame où les valeurs manquantes sont imputés selon les
paramètres passés en argument.

#### Parameters
- train_df: DataFrame
- columns_def: dictionnaire où chaque clef est un nom de colonne du DataFrame et où
les clefs / valeurs sont les paramètres pour l'imputer

#### Return
Renvoi un DataFrame où les *na* des colonnes passées en argument sont remplis
