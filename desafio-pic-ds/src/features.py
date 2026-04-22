import pandas as pd
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

class DateFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Extrator customizado de features de data usando Scikit-Learn.
    """
    def __init__(self, date_column: str):
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.date_column in X_out.columns:
            # Garante que seja datetime
            if not pd.api.types.is_datetime64_any_dtype(X_out[self.date_column]):
                X_out[self.date_column] = pd.to_datetime(X_out[self.date_column])

            X_out[f'{self.date_column}_month'] = X_out[self.date_column].dt.month
            X_out[f'{self.date_column}_dayofweek'] = X_out[self.date_column].dt.dayofweek
            X_out[f'{self.date_column}_is_weekend'] = X_out[self.date_column].dt.dayofweek.isin([5, 6]).astype(int)
        return X_out

def get_preprocessing_pipeline(
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> ColumnTransformer:
    """
    Constrói a pipeline de preprocessamento do Scikit-Learn.

    Args:
        categorical_cols (List[str]): Lista de variáveis categóricas.
        numerical_cols (List[str]): Lista de variáveis numéricas.

    Returns:
        ColumnTransformer: O processador pré-configurado.
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor

def create_stratified_sample(df: pd.DataFrame, n_samples: int = 50000, stratify_col: str = 'area_planejamento') -> pd.DataFrame:
    """
    Cria uma amostra estratificada do DataFrame para garantir representatividade territorial.

    Args:
        df: DataFrame original
        n_samples: Tamanho da amostra desejado
        stratify_col: Coluna para estratificar

    Returns:
        DataFrame amostrado.
    """
    if len(df) <= n_samples:
        return df.copy()

    # Calcula as proporções
    proportions = df[stratify_col].value_counts(normalize=True)

    sampled_dfs = []
    for category, prop in proportions.items():
        n_cat_samples = int(n_samples * prop)
        cat_df = df[df[stratify_col] == category]
        sampled_dfs.append(cat_df.sample(n=min(n_cat_samples, len(cat_df)), random_state=42))

    return pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def temporal_train_test_split(df: pd.DataFrame, date_col: str, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Faz o split temporal dos dados evitando data leakage.

    Args:
        df (pd.DataFrame): O DataFrame a ser dividido.
        date_col (str): Coluna contendo a data.
        split_date (str): A data (YYYY-MM-DD) para fazer a divisão. Treino < split_date, Teste >= split_date

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Treino, Teste)
    """
    df_sorted = df.sort_values(by=date_col)
    train_mask = df_sorted[date_col] < pd.to_datetime(split_date)

    train = df_sorted[train_mask].copy()
    test = df_sorted[~train_mask].copy()

    return train, test
