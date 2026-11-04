"""Catálogo de features e lista de colunas excluídas, com a justificativa de cada decisão.

Conversão de `2. Pre-processamento.ipynb`, seção 5. As evidências apontam para as seções
de `1. EDA.ipynb`.
"""
import numpy as np
import pandas as pd

from src.common import config

METADADOS_FAMILIAS = pd.DataFrame([
    ("rede", "Gold", "ano-alvo", "seção 8", "baixo"),
    ("territorio_categorico", "Gold / Base dos Dados", "fixo", "seções 8, 12 e 14", "pode mascarar políticas estaduais; avaliado com e sem território"),
    ("territorio_numerico", "Base dos Dados", "fixo", "seção 8", "baixo"),
    ("historico", "Gold", "ano-alvo - 1", "seções 7, 8, 9 e 14", "só existe no alvo 2024 em diante"),
    ("socioeconomico", "Atlas 2010 / IBGE", "2010; até ano-alvo - 1", "seções 11 e 12", "Atlas defasado; alta redundância"),
    ("infraestrutura", "Censo Escolar", "ano-alvo - 1", "seção 11", "baixo"),
    ("docentes_turmas", "Censo Escolar / INEP", "2022; ano-alvo - 1", "seção 11", "baixo"),
    ("fluxo_escolar", "INEP", "2022", "seções 11 e 12", "pode refletir desempenho anterior"),
    ("financiamento", "Fundeb / SIOPE", "ano-alvo - 1", "seção 11", "associação fraca; municipal e estadual diferem"),
    ("escola", "Gold", "ano-alvo", "seção 8", "supõe lista de alunos prévia à prova"),
    ("metas", "Gold", "definida antes do ano-alvo", "seção 10", "só no modelo municipal"),
], columns=["familia", "fonte", "referencia", "evidencia_eda", "risco"])

EXCLUIDAS = pd.DataFrame([
    ("proficiencia", "vazamento", "define o alvo (EDA, seção 13)"),
    ("media_portugues, proporcao_aluno_nivel_*, taxa_alfabetizacao do mesmo ano", "vazamento", "agregam o alvo (EDA, seção 13)"),
    ("presenca, preenchimento_caderno e IDs", "vazamento", "conhecidas só após a prova (EDA, seção 13)"),
    ("peso_aluno", "vazamento", "criado após a prova; usado só como sample_weight (EDA, seções 6 e 13)"),
    ("gap_meta_ano, gap_ate_2030, atingiu_meta_ano", "vazamento", "revelam o alvo municipal (EDA, seção 13)"),
    ("médias de escola ou município do mesmo ano", "vazamento", "incluem o alvo (EDA, seção 13)"),
    ("id_aluno", "identificação", "identificador individual, sem conteúdo preditivo e sem uso fora do treino"),
    ("id_escola", "não estável", "recodificado a cada ano e sem correspondência com o Censo (EDA, seção 5)"),
    ("id_municipio", "generalização", "não se aplica a municípios fora do treino; o histórico já cobre esse sinal"),
    ("nome_mesorregiao", "validação", "é a chave de separação entre treino e teste (EDA, seção 8)"),
    ("ano", "generalização", "não se projeta para anos futuros; usado só para filtro"),
    ("serie", "sem variância", "o indicador avalia apenas o 2º ano"),
    ("caderno", "vazamento", "versão da prova, definida só na aplicação"),
    ("pct_internet_alunos, horas_aula_2ano, despesa_professor_aluno", "sem sinal", "associação próxima de zero (EDA, seção 11)"),
    ("proporções por nível do ano anterior", "indisponível", "não existem em 2023 (EDA, seção 6)"),
], columns=["coluna", "motivo", "justificativa"])


def construir_catalogo() -> pd.DataFrame:
    """Uma linha por feature, com fonte, risco, transformação e o modelo que a usa."""
    familias = {**config.FAMILIAS_ALUNO, "metas": config.FAMILIAS["metas"]}
    df = pd.DataFrame([{"feature": f, "familia": familia} for familia, colunas in familias.items() for f in colunas]).merge(METADADOS_FAMILIAS, on="familia")
    df["tipo"] = np.where(df["feature"].isin(config.CATEGORICAS_ALUNO), "categórica", "numérica")
    df["transformacao"] = np.select([df["feature"].isin(config.CATEGORICAS_ALUNO), df["feature"].isin(config.NUMERICAS_LOG_ALUNO)],
                                    ["ausente + one-hot", "mediana + log1p + padronização"], "mediana + padronização")
    df["modelo_aluno"] = df["feature"].isin(config.FEATURES_ALUNO)
    df["modelo_municipio"] = df["feature"].isin(config.FEATURES)
    return df
