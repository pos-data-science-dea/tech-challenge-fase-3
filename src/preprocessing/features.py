"""Engenharia de atributos: contexto por município, rede e ano-alvo.

Conversão de `2. Pre-processamento.ipynb`, seção 4. Resultados, participação e
metas entram defasados, para evitar vazamento do alvo.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.common import config


def media_vizinhos(valores, coords, k: int) -> np.ndarray:
    """Média do valor nos k vizinhos mais próximos, excluindo o próprio município."""
    valores, coords = np.asarray(valores, dtype=float), np.asarray(coords, dtype=float)
    validos = np.flatnonzero(~np.isnan(valores) & ~np.isnan(coords).any(axis=1))
    com_coord = ~np.isnan(coords).any(axis=1)
    vizinhos = cKDTree(coords[validos]).query(coords[com_coord], k=k + 1)[1]
    origem = validos[vizinhos]
    proprio = origem == np.flatnonzero(com_coord)[:, None]
    proprio[~proprio.any(axis=1), -1] = True
    resultado = np.full(len(valores), np.nan)
    resultado[com_coord] = np.ma.array(valores[origem], mask=proprio).mean(axis=1).filled(np.nan)
    return resultado


def suavizar(valor, n, referencia, k: int):
    """Aproxima da referência as taxas de municípios com poucos alunos avaliados."""
    return (n * valor + k * referencia) / (n + k)


def reduzir_numericas(df: pd.DataFrame) -> pd.DataFrame:
    colunas = df.select_dtypes("float64").columns
    return df.astype({c: "float32" for c in colunas})


def _adicionar_externas(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    df = (df
          .merge(f["atlas"], on="id_municipio", how="left")
          .merge(f["populacao"].drop(columns="ano_ref"), on=["ano_alvo", "id_municipio"], how="left")
          .merge(f["pib"].drop(columns="ano_ref"), on=["ano_alvo", "id_municipio"], how="left")
          .merge(f["censo"].drop(columns="ano_ref"), on=["ano_alvo", "id_municipio", "rede_id"], how="left")
          .merge(f["inep"], on=["id_municipio", "rede_id"], how="left")
          .merge(f["fundeb_municipal"].drop(columns="ano_ref").rename(columns={"invest_aluno_fundamental": "invest_municipal"}),
                 on=["ano_alvo", "id_municipio"], how="left")
          .merge(f["fundeb_estadual"].drop(columns="ano_ref").rename(columns={"invest_aluno_fundamental": "invest_estadual"}),
                 on=["ano_alvo", "sigla_uf"], how="left"))
    investimento = np.where(df["rede_id"] == config.REDE_MUNICIPAL, df["invest_municipal"], df["invest_estadual"])
    df["invest_aluno_fundamental"] = pd.Series(investimento, index=df.index).where(lambda s: s > 0)  # zero equivale a ausente
    return df.drop(columns=["invest_municipal", "invest_estadual"])


def _adicionar_historico(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    lag_indicador = (f["indicador"][f["indicador"]["rede_id"].isin(config.REDES)]
                     .assign(ano_alvo=lambda d: d["ano"] + 1)
                     .rename(columns={"taxa_alfabetizacao": "taxa_lag", "media_portugues": "media_portugues_lag"})
                     [["ano_alvo", "id_municipio", "rede_id", "taxa_lag", "media_portugues_lag"]])
    lag_participacao = f["participacao"].assign(ano_alvo=lambda d: d["ano"] + 1,
                                                taxa_presenca_lag=lambda d: 100 * d["n_presentes"] / d["n_total"],
                                                n_avaliados_lag=lambda d: d["n_avaliados"].astype(float))
    df = (df
          .merge(lag_indicador, on=["ano_alvo", "id_municipio", "rede_id"], how="left")
          .merge(lag_participacao[["ano_alvo", "id_municipio", "rede_id", "taxa_presenca_lag", "n_avaliados_lag"]],
                 on=["ano_alvo", "id_municipio", "rede_id"], how="left"))
    df["taxa_lag_uf"] = df.groupby(["ano_alvo", "rede_id", "sigla_uf"])["taxa_lag"].transform("mean")
    df["taxa_lag_suavizada"] = suavizar(df["taxa_lag"], df["n_avaliados_lag"], df["taxa_lag_uf"], config.K_SUAVIZACAO).fillna(df["taxa_lag"])
    df["taxa_lag_vizinhos"] = np.nan
    for _, d in df.groupby(["ano_alvo", "rede_id"]):
        if d["taxa_lag"].notna().any():
            df.loc[d.index, "taxa_lag_vizinhos"] = media_vizinhos(d["taxa_lag"], d[["lon", "lat"]], config.K_VIZINHOS)
    return df


def _adicionar_metas(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    metas = (f["meta"][f["meta"]["nivel_geografico"] == "municipio"]
             .sort_values("ano_referencia").drop_duplicates(["local_id", "ano_meta"], keep="last")
             .rename(columns={"local_id": "id_municipio", "ano_meta": "ano_alvo", "valor_meta": "meta_ano"})
             [["ano_alvo", "id_municipio", "meta_ano"]])
    df = df.merge(metas.assign(rede_id=config.REDE_MUNICIPAL), on=["ano_alvo", "id_municipio", "rede_id"], how="left")
    df["esforco_meta"] = df["meta_ano"] - df["taxa_lag"]
    return df


def construir_contexto(f: dict) -> pd.DataFrame:
    """Monta a tabela município x rede x ano-alvo com todas as features."""
    municipios = f["municipio"].sort_values("id_municipio")  # a ordem de leitura do BigQuery não é garantida
    df = (pd.MultiIndex.from_product([config.ANOS_ALVO, municipios["id_municipio"], config.REDES],
                                     names=["ano_alvo", "id_municipio", "rede_id"]).to_frame(index=False)
          .merge(municipios[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao"]], on="id_municipio")
          .merge(f["territorio"], on="id_municipio", how="left")
          .merge(f["rede"], on="rede_id", how="left"))
    df = _adicionar_externas(df, f)
    df = _adicionar_historico(df, f)
    df = _adicionar_metas(df, f)
    return reduzir_numericas(df)
