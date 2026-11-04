"""Funções de métricas usadas na avaliação do modelo final."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error

from src.common import config


def regressao(y, previsto) -> dict:
    return {"rmse": root_mean_squared_error(y, previsto), "mae": mean_absolute_error(y, previsto), "r2": r2_score(y, previsto)}


def meta(y_meta, folga) -> dict:
    """Trata a diferença entre taxa prevista e meta como placar do classificador."""
    atinge = folga >= 0
    return {"auc": roc_auc_score(y_meta, folga), "acuracia_balanceada": balanced_accuracy_score(y_meta, atinge),
            "precisao_nao_atinge": precision_score(y_meta, atinge, pos_label=0), "recall_nao_atinge": recall_score(y_meta, atinge, pos_label=0)}


def bootstrap_grupos(grupos, funcao, repeticoes: int = config.BOOTSTRAP, seed: int = config.SEED) -> np.ndarray:
    """Intervalo de 95% reamostrando mesorregiões, a unidade independente da validação."""
    rng = np.random.default_rng(seed)
    posicoes = pd.Series(np.arange(len(grupos))).groupby(np.asarray(grupos)).apply(lambda s: s.values)
    valores = []
    for _ in range(repeticoes):
        indices = np.concatenate(posicoes.values[rng.integers(0, len(posicoes), len(posicoes))])
        valores.append(funcao(indices))
    return np.percentile(valores, [2.5, 50, 97.5])


def variavel_original(nome: str) -> str:
    """Soma one-hot e indicadores de ausência na variável de origem."""
    nome = nome.replace("ausente_", "")
    return next((c for c in config.CATEGORICAS_ALUNO if nome.startswith(c + "_")), nome)
