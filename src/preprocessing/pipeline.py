"""Gera as bases, os pré-processadores e o catálogo de features dos dois modelos.

Uso:
    python -m src.preprocessing.pipeline
"""
import joblib
from google.cloud import bigquery

from src.common import config
from src.preprocessing import bases, catalogo, extracao, features
from src.preprocessing.preprocessador import construir_preprocessador


def run() -> None:
    fontes = extracao.extrair(bigquery.Client(project=config.PROJECT_ID))
    contexto = features.construir_contexto(fontes)
    municipio = bases.base_principal(contexto, fontes)
    particoes = bases.separar_mesorregioes(contexto, fontes)
    municipio["split"] = municipio[config.GRUPO].map(particoes)
    aluno = bases.base_aluno(contexto, fontes)
    aluno["split"] = aluno[config.GRUPO].map(particoes)
    escoragem = bases.base_escoragem(contexto)
    temporal = bases.base_temporal(contexto, fontes)
    bases.validar(municipio, escoragem, fontes)
    bases.validar_aluno(aluno)

    config.DIR_DADOS.mkdir(parents=True, exist_ok=True)
    municipio.to_parquet(config.ARQ_BASE, index=False)
    escoragem.to_parquet(config.ARQ_ESCORAGEM, index=False)
    temporal.to_parquet(config.ARQ_BASE_2023, index=False)
    aluno.to_parquet(config.ARQ_BASE_ALUNO, index=False)
    catalogo.construir_catalogo().to_csv(config.ARQ_CATALOGO, index=False)
    catalogo.EXCLUIDAS.to_csv(config.ARQ_EXCLUIDAS, index=False)
    joblib.dump(construir_preprocessador(), config.ARQ_PREPROCESSADOR)
    joblib.dump(construir_preprocessador(config.NUMERICAS_ALUNO, config.NUMERICAS_LOG_ALUNO, config.CATEGORICAS_ALUNO), config.ARQ_PREPROCESSADOR_ALUNO)
    print(f"Alunos: {aluno.shape} | Base 2024: {municipio.shape} | Escoragem 2025: {escoragem.shape} | Base 2023: {temporal.shape}")
    print(municipio["split"].value_counts().to_string())


if __name__ == "__main__":
    run()
