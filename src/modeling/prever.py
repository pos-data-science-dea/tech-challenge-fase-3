"""Prevê a taxa de 2025 e classifica risco e alerta de meta por município.

Regras de `5. Aplicacao Estrategica.ipynb`: o risco usa o cenário sem território,
e o alerta de meta só é firme quando os dois cenários concordam.

Uso:
    python -m src.modeling.prever
"""
import joblib
import numpy as np
import pandas as pd

from src.common import config
from src.modeling import modelo as m


def _alerta(probabilidade) -> np.ndarray:
    return np.select([probabilidade <= config.PROB_BAIXA, probabilidade >= config.PROB_ALTA], ["Provável não atingir", "Provável atingir"], "Incerto")


def run() -> pd.DataFrame:
    df = pd.read_parquet(config.ARQ_ESCORAGEM)
    historico = pd.read_parquet(config.ARQ_BASE, columns=["id_municipio", "taxa_lag"]).rename(columns={"taxa_lag": "taxa_2023"})
    residuos = pd.read_parquet(config.ARQ_RESIDUOS)
    producao, sem_territorio = joblib.load(config.ARQ_MODELO), joblib.load(config.ARQ_MODELO_SEM_TERRITORIO)

    df["faixa"] = m.faixa_matriculas(df)
    df["taxa_2024"] = df["taxa_lag"].astype(float)
    df["meta_2025"] = df["meta_ano"].astype(float)
    df["taxa_prevista_2025"] = np.clip(producao.predict(df[config.FEATURES]), 0, 100)
    df["prevista_sem_territorio"] = np.clip(sem_territorio.predict(df[config.FEATURES]), 0, 100)
    df["folga_meta_2025"] = df["taxa_prevista_2025"] - df["meta_ano"].astype(float)
    df["atinge_meta_previsto"] = df["folga_meta_2025"] >= 0

    # Mesmo formato gerado pelo notebook 3.
    df[["id_municipio", "nome_municipio", "sigla_uf", "taxa_lag", "taxa_prevista_2025", "meta_ano", "folga_meta_2025", "atinge_meta_previsto"]].to_csv(config.ARQ_PREVISOES, index=False)

    df["prob_abaixo_limiar"] = 1 - m.prob_acima(df["prevista_sem_territorio"], config.LIMIAR_TAXA, df["faixa"], residuos["faixa"], residuos["residuo_sem_territorio"].values)
    df["nivel_risco"] = np.select([df["prob_abaixo_limiar"] >= config.PROB_ALTA, df["prob_abaixo_limiar"] > config.PROB_BAIXA], ["Alto", "Moderado"], "Baixo")
    df["prob_meta_producao"] = m.prob_acima(df["taxa_prevista_2025"], df["meta_2025"], df["faixa"], residuos["faixa"], residuos["residuo_producao"].values)
    df["prob_meta_sem_territorio"] = m.prob_acima(df["prevista_sem_territorio"], df["meta_2025"], df["faixa"], residuos["faixa"], residuos["residuo_sem_territorio"].values)
    alerta_producao, alerta_sem_territorio = _alerta(df["prob_meta_producao"]), _alerta(df["prob_meta_sem_territorio"])
    df["alerta_meta"] = np.where(alerta_producao == alerta_sem_territorio, alerta_sem_territorio, "Incerto")
    df["criancas_nao_alfabetizadas"] = df["n_avaliados_lag"].astype(float) * (100 - df["prevista_sem_territorio"]) / 100
    df["amostra_pequena"] = df["n_avaliados_lag"].astype(float) < config.MIN_AVALIADOS
    df = df.merge(historico, on="id_municipio", how="left")
    df["abaixo_dois_anos"] = (df["taxa_2023"].astype(float) < config.LIMIAR_TAXA) & (df["taxa_2024"] < config.LIMIAR_TAXA)

    saida = df[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "lon", "lat", "n_avaliados_lag", "taxa_2023", "taxa_2024", "meta_2025", "taxa_prevista_2025",
                "prevista_sem_territorio", "prob_abaixo_limiar", "nivel_risco", "abaixo_dois_anos", "amostra_pequena", "criancas_nao_alfabetizadas",
                "prob_meta_producao", "prob_meta_sem_territorio", "alerta_meta"]]
    saida.to_csv(config.ARQ_CENARIOS, index=False)
    print("Risco:", saida["nivel_risco"].value_counts().to_dict(), "| Metas:", saida["alerta_meta"].value_counts().to_dict())
    return saida


if __name__ == "__main__":
    run()
