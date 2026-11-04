"""Bases dos modelos e separação em treino, validação e teste.

Conversão de `2. Pre-processamento.ipynb`, seções 6 e 7. A separação é feita por
mesorregião, com pesos pelo número de alunos avaliados, a mesma regra usada no notebook.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import config


def base_principal(contexto: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Rede Municipal em 2024, somente municípios com meta definida."""
    resultados = f["resultados"]
    alvo = resultados[(resultados["rede_id"] == config.REDE_MUNICIPAL) & (resultados["ano"] == config.ANO_PRINCIPAL)]
    df = (contexto[(contexto["ano_alvo"] == config.ANO_PRINCIPAL) & (contexto["rede_id"] == config.REDE_MUNICIPAL)]
          .merge(alvo[["id_municipio", "taxa_alfabetizacao", "atingiu_meta_ano"]], on="id_municipio"))
    df = df[df["meta_ano"].notna()].copy()
    df["atingiu_meta"] = df["atingiu_meta_ano"].astype(int)
    return df.drop(columns="atingiu_meta_ano")


def base_aluno(contexto: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Alunos avaliados das redes Estadual e Municipal, com o contexto de cada ano."""
    df = (f["aluno"].rename(columns={"ano": "ano_alvo"})
          .merge(contexto.drop(columns=config.FAMILIAS["metas"]), on=["ano_alvo", "id_municipio", "rede_id"], how="left"))
    df["alunos_previstos_escola"] = df["alunos_previstos_escola"].astype("float32")
    df["alfabetizado"] = df["alfabetizado"].astype("int8")
    df["peso_amostral"] = df["peso_aluno"].clip(upper=config.PESO_MAXIMO).astype("float32")
    # A ordem de leitura do BigQuery não é garantida, e a amostragem depende da posição das linhas.
    ordem = ["ano_alvo", "id_municipio", "rede_id", "id_escola", "alfabetizado", "peso_amostral"]
    return df.drop(columns=["peso_aluno", "nome_municipio", "nome_uf"], errors="ignore").sort_values(ordem, kind="stable").reset_index(drop=True)


def base_escoragem(contexto: pd.DataFrame) -> pd.DataFrame:
    """Metas de 2025, a partir dos resultados de 2024."""
    return contexto[(contexto["ano_alvo"] == config.ANO_ESCORAGEM) & (contexto["rede_id"] == config.REDE_MUNICIPAL)
                    & contexto["meta_ano"].notna() & contexto["taxa_lag"].notna()].copy()


def base_temporal(contexto: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Alvo de 2023, usado só no teste temporal. Municípios com alunos avaliados naquele ano."""
    participacao = f["participacao"]
    com_alunos = participacao[(participacao["ano"] == config.ANO_TREINO_TEMPORAL) & (participacao["rede_id"] == config.REDE_MUNICIPAL)
                              & (participacao["n_avaliados"] > 0)][["id_municipio"]]
    indicador = f["indicador"]
    taxa = indicador[(indicador["ano"] == config.ANO_TREINO_TEMPORAL) & (indicador["rede_id"] == config.REDE_MUNICIPAL)][["id_municipio", "taxa_alfabetizacao"]]
    df = (contexto[(contexto["ano_alvo"] == config.ANO_TREINO_TEMPORAL) & (contexto["rede_id"] == config.REDE_MUNICIPAL)]
          .merge(com_alunos, on="id_municipio").merge(taxa, on="id_municipio"))
    df[["meta_ano", "esforco_meta"]] = np.nan
    return df


def separar_mesorregioes(contexto: pd.DataFrame, f: dict) -> pd.Series:
    """Distribui mesorregiões entre as partições, com todas as UFs no treino."""
    participacao = f["participacao"]
    alunos = (participacao[(participacao["ano"] == config.ANO_PRINCIPAL) & (participacao["n_avaliados"] > 0)]
              .rename(columns={"n_avaliados": "alunos"})[["id_municipio", "rede_id", "alunos"]])
    grupos = (contexto[contexto["ano_alvo"] == config.ANO_PRINCIPAL].merge(alunos, on=["id_municipio", "rede_id"])
              .groupby(["sigla_uf", config.GRUPO], as_index=False).agg(alunos=("alunos", "sum"))
              .sample(frac=1, random_state=config.SEED).reset_index(drop=True))
    alvos = {"treino": 1 - config.PROP_TESTE - config.PROP_VALIDACAO, "validacao": config.PROP_VALIDACAO, "teste": config.PROP_TESTE}
    grupos = grupos.sort_values("alunos", ascending=False).reset_index(drop=True)
    grupos["split"] = None
    grupos.loc[grupos.groupby("sigla_uf").head(1).index, "split"] = "treino"
    acumulado = {k: grupos.loc[grupos["split"] == k, "alunos"].sum() for k in alvos}
    total = grupos["alunos"].sum()
    for i in grupos.index[grupos["split"].isna()]:
        destino = max(alvos, key=lambda k: alvos[k] * total - acumulado[k])
        grupos.loc[i, "split"] = destino
        acumulado[destino] += grupos.loc[i, "alunos"]
    return grupos.set_index(config.GRUPO)["split"]


def validar(municipio: pd.DataFrame, escoragem: pd.DataFrame, f: dict) -> None:
    """Interrompe a execução se houver indício de vazamento ou falha de cobertura."""
    assert not config.PROIBIDAS & set(config.FEATURES), "Coluna proibida entre as features"
    for nome in ["populacao", "pib", "censo", "fundeb_municipal", "fundeb_estadual"]:
        assert (f[nome]["ano_ref"] < f[nome]["ano_alvo"]).all(), f"Fonte {nome} posterior ao ano-alvo"
    assert municipio["split"].notna().all(), "Município sem partição"
    assert municipio.groupby(config.GRUPO)["split"].nunique().max() == 1, "Mesorregião em mais de uma partição"
    assert set(config.FEATURES) <= set(municipio.columns) and set(config.FEATURES) <= set(escoragem.columns), "Feature ausente nas bases"
    treino = municipio[municipio["split"] == "treino"]
    for coluna in [c for c in config.FEATURES if c not in config.CATEGORICAS]:
        d = treino[treino[coluna].notna()]
        auc = roc_auc_score(d[config.ALVO_META], d[coluna].astype(float))
        assert abs(auc - .5) + .5 < .9, f"{coluna} quase determina o alvo (AUC {auc:.3f})"


def validar_aluno(aluno: pd.DataFrame) -> None:
    """Interrompe a execução se houver indício de vazamento na base de alunos."""
    assert not config.PROIBIDAS & set(config.FEATURES_ALUNO), "Coluna proibida entre as features do aluno"
    assert set(config.FEATURES_ALUNO) <= set(aluno.columns), "Feature ausente na base de alunos"
    assert aluno["split"].notna().all(), "Aluno sem partição"
    assert aluno.groupby(config.GRUPO)["split"].nunique().max() == 1, "Mesorregião em mais de uma partição"
    treino = aluno[(aluno["ano_alvo"] == config.ANO_PRINCIPAL) & (aluno["split"] == "treino")].sample(config.AMOSTRA_ALUNO, random_state=config.SEED)
    for coluna in [c for c in config.FEATURES_ALUNO if c not in config.CATEGORICAS_ALUNO]:
        d = treino[treino[coluna].notna()]
        auc = roc_auc_score(d[config.ALVO_ALUNO], d[coluna].astype(float))
        assert abs(auc - .5) + .5 < .9, f"{coluna} quase determina o alvo do aluno (AUC {auc:.3f})"
