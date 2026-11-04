"""Pré-processador dos modelos: imputação, log, padronização e one-hot.

Fica sem ajuste no arquivo salvo: como primeira etapa do Pipeline, é ajustado só
com os dados de treino de cada dobra.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.common import config


def construir_preprocessador(numericas: list = config.NUMERICAS, numericas_log: list = config.NUMERICAS_LOG, categoricas: list = config.CATEGORICAS) -> ColumnTransformer:
    imputacao = SimpleImputer(strategy="median")
    return ColumnTransformer([
        ("num", Pipeline([("imputacao", imputacao), ("escala", StandardScaler())]), numericas),
        ("log", Pipeline([("imputacao", imputacao), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("escala", StandardScaler())]), numericas_log),
        ("ausente", MissingIndicator(features="missing-only", error_on_new=False), numericas + numericas_log),
        ("cat", Pipeline([("imputacao", SimpleImputer(strategy="constant", fill_value="ausente")),
                          ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas),
    ])
