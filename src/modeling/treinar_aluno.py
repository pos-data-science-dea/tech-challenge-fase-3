"""Treina o modelo de alfabetização por aluno com toda a base de 2024 e salva o artefato.

O desempenho é medido em `src.evaluation.avaliar_aluno`, com um ajuste que não usa o teste.

Uso:
    python -m src.modeling.treinar_aluno
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.common import config


def criar_modelo(preprocessador) -> Pipeline:
    """Modelo final: LightGBM com pré-processamento integrado."""
    return Pipeline([("prep", clone(preprocessador)), ("modelo", lgb.LGBMClassifier(**config.LIGHTGBM_ALUNO))])


def carregar_base(ano: int = config.ANO_PRINCIPAL, colunas_extras: tuple = ()) -> pd.DataFrame:
    colunas = [*config.FEATURES_ALUNO, config.ALVO_ALUNO, config.PESO_ALUNO, config.GRUPO, "split", "ano_alvo", *colunas_extras]
    df = pd.read_parquet(config.ARQ_BASE_ALUNO, columns=colunas)
    df = df[df["ano_alvo"] == ano].drop(columns="ano_alvo")
    inteiras = df.select_dtypes("Int64").columns
    df[inteiras] = df[inteiras].astype("float32")
    return df


def ajustar(modelo: Pipeline, df: pd.DataFrame) -> Pipeline:
    return clone(modelo).fit(df[config.FEATURES_ALUNO], df[config.ALVO_ALUNO].values, modelo__sample_weight=df[config.PESO_ALUNO].values)


def prever(modelo: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(df[config.FEATURES_ALUNO])[:, 1]


def run() -> None:
    modelo = criar_modelo(joblib.load(config.ARQ_PREPROCESSADOR_ALUNO))
    df = carregar_base()
    joblib.dump(ajustar(modelo, df), config.ARQ_MODELO_ALUNO)
    print(f"Modelo de aluno treinado com {len(df)} alunos")


if __name__ == "__main__":
    run()
