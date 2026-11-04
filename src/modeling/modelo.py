"""Definição do modelo final e funções compartilhadas por treino e avaliação."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.common import config


def criar_modelo(preprocessador) -> Pipeline:
    """Modelo final: Lasso com pré-processamento integrado."""
    return Pipeline([("prep", clone(preprocessador)), ("modelo", Lasso(alpha=config.LASSO_ALPHA, max_iter=config.LASSO_MAX_ITER))])


def sem_colunas(preprocessador, remover, sem_indicadores: bool = False):
    novo = clone(preprocessador)
    novo.transformers = [(n, t, [c for c in colunas if c not in remover]) for n, t, colunas in novo.transformers
                         if not (sem_indicadores and n == "ausente")]
    return novo


def criar_modelo_sem_territorio(preprocessador) -> Pipeline:
    """Cenário de sensibilidade: não repete em 2025 os choques estaduais de 2024, que já estão no histórico."""
    return criar_modelo(sem_colunas(preprocessador, config.TERRITORIO, True))


def criar_modelo_sem_historico(preprocessador) -> Pipeline:
    """Teste temporal: histórico e metas não existem para o alvo de 2023."""
    return criar_modelo(sem_colunas(preprocessador, config.FAMILIAS["historico"] + config.FAMILIAS["metas"], True))


def faixa_matriculas(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["matriculas_anos_iniciais"], config.FAIXAS_MATRICULAS, labels=config.ROTULOS_MATRICULAS, right=False).astype(str)


def residuos_oof(modelo: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Erros fora da amostra, sempre em mesorregiões que não participaram do ajuste."""
    y = df[config.ALVO].astype(float)
    previsto = cross_val_predict(clone(modelo), df[config.FEATURES], y, groups=df[config.GRUPO], cv=GroupKFold(n_splits=config.DOBRAS))
    return y.values - previsto


def prob_acima(previsto, limite, faixas, faixas_ref, residuos) -> np.ndarray:
    """Probabilidade de a taxa superar o limite, usando os erros da mesma faixa de porte."""
    distancia, faixas, faixas_ref = np.asarray(limite, dtype=float) - np.asarray(previsto, dtype=float), np.asarray(faixas), np.asarray(faixas_ref)
    probabilidade = np.full(len(distancia), np.nan)
    for faixa in np.unique(faixas):
        mascara = faixas == faixa
        probabilidade[mascara] = (residuos[faixas_ref == faixa][:, None] >= distancia[mascara]).mean(axis=0)
    return probabilidade


def nomes_saida(preprocessador) -> list:
    return [c.split("__")[1].replace("missingindicator_", "ausente_") for c in preprocessador.get_feature_names_out()]
