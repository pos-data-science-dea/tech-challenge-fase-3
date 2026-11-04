"""Treina o modelo final com toda a base de 2024 e salva os artefatos.

O desempenho é medido em `src.evaluation.avaliar`, com um ajuste que não usa o teste.

Uso:
    python -m src.modeling.treinar
"""
import joblib
import pandas as pd

from src.common import config
from src.modeling import modelo as m


def run() -> None:
    df = pd.read_parquet(config.ARQ_BASE)
    preprocessador = joblib.load(config.ARQ_PREPROCESSADOR)
    X, y = df[config.FEATURES], df[config.ALVO].astype(float)
    producao, sem_territorio = m.criar_modelo(preprocessador), m.criar_modelo_sem_territorio(preprocessador)

    residuos = pd.DataFrame({"id_municipio": df["id_municipio"], "faixa": m.faixa_matriculas(df),
                             "residuo_producao": m.residuos_oof(producao, df), "residuo_sem_territorio": m.residuos_oof(sem_territorio, df)})
    producao.fit(X, y)
    sem_territorio.fit(X, y)

    joblib.dump(producao, config.ARQ_MODELO)
    joblib.dump(sem_territorio, config.ARQ_MODELO_SEM_TERRITORIO)
    residuos.to_parquet(config.ARQ_RESIDUOS, index=False)
    coeficientes = pd.Series(producao.named_steps["modelo"].coef_, index=m.nomes_saida(producao.named_steps["prep"]))
    print(f"Modelo treinado com {len(df)} municípios | Termos ativos: {(coeficientes != 0).sum()} de {len(coeficientes)}")


if __name__ == "__main__":
    run()
