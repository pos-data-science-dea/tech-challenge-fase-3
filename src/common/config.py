"""Configuração do projeto, parametrizável via variáveis de ambiente.

Reúne os parâmetros definidos nos notebooks e os hiperparâmetros dos modelos finais
(`notebooks/3. Modelagem.ipynb` e `notebooks/4. Modelo de aluno.ipynb`). Os diretórios podem ser trocados com `DATA_DIR`
e `REPORTS_DIR`, por exemplo para testar sem sobrescrever artefatos.
"""
import os
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DIR_DADOS = Path(os.environ.get("DATA_DIR", RAIZ / "data" / "processed"))
DIR_REPORTS = Path(os.environ.get("REPORTS_DIR", RAIZ / "reports"))
DIR_AVALIACAO = DIR_REPORTS / "avaliacao"

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "tech-challenge-fase-2-505123")
GOLD = f"{PROJECT_ID}.{os.environ.get('GOLD_DATASET', 'gold')}"
BASE_DOS_DADOS = os.environ.get("BASE_DOS_DADOS_PROJECT", "basedosdados")

ANOS_ALVO = [2023, 2024, 2025]
ANO_TREINO_TEMPORAL = 2023
ANO_PRINCIPAL = 2024
ANO_ESCORAGEM = 2025
ANO_INEP = 2022
ANO_ATLAS = 2010
REDE_ESTADUAL = "2"
REDE_MUNICIPAL = "3"
REDES = [REDE_ESTADUAL, REDE_MUNICIPAL]

K_SUAVIZACAO = 30
K_VIZINHOS = 8
SEED = 42
PROP_TESTE = .15
PROP_VALIDACAO = .15

ALVO = "taxa_alfabetizacao"
ALVO_META = "atingiu_meta"
GRUPO = "nome_mesorregiao"
DOBRAS = 5
BOOTSTRAP = 2000
REPETICOES_PERMUTACAO = 10
AMOSTRA_SHAP = 300

# Melhor busca do notebook 3: Lasso, com menor RMSE na validação cruzada por mesorregião.
LASSO_ALPHA = 0.06672367170464207
LASSO_MAX_ITER = 20000

FAIXAS_MATRICULAS = [0, 200, 600, 2000, np.inf]
ROTULOS_MATRICULAS = ["<200", "200-599", "600-1999", "2000+"]

# Regras de uso definidas no notebook 5.
LIMIAR_TAXA = 50
PROB_ALTA = .7
PROB_BAIXA = .3
MIN_AVALIADOS = 30

FAMILIAS = {
    "territorio_categorico": ["sigla_uf", "nome_regiao", "capital_uf", "amazonia_legal"],
    "territorio_numerico": ["lat", "lon"],
    "historico": ["taxa_lag", "taxa_lag_suavizada", "media_portugues_lag", "taxa_presenca_lag", "n_avaliados_lag",
                  "taxa_lag_uf", "taxa_lag_vizinhos"],
    "socioeconomico": ["idhm", "idhm_e", "renda_pc", "indice_gini", "prop_pobreza_criancas", "expectativa_anos_estudo",
                       "populacao", "pib_per_capita"],
    "infraestrutura": ["escolas_anos_iniciais", "matriculas_anos_iniciais", "pct_escolas_rurais", "pct_biblioteca", "pct_esgoto_rede"],
    "docentes_turmas": ["alunos_por_turma", "alunos_por_docente", "alunos_turma_2ano", "pct_docente_superior",
                        "pct_docente_adequado", "pct_docente_baixa_regularidade"],
    "fluxo_escolar": ["distorcao_idade_serie", "taxa_aprovacao", "taxa_abandono"],
    "financiamento": ["invest_aluno_fundamental"],
    "metas": ["meta_ano", "esforco_meta"],
}
FEATURES = [f for colunas in FAMILIAS.values() for f in colunas]
CATEGORICAS = FAMILIAS["territorio_categorico"]
TERRITORIO = FAMILIAS["territorio_categorico"] + FAMILIAS["territorio_numerico"]
NUMERICAS_LOG = ["n_avaliados_lag", "renda_pc", "populacao", "pib_per_capita", "escolas_anos_iniciais", "matriculas_anos_iniciais",
                 "alunos_por_turma", "alunos_por_docente", "invest_aluno_fundamental"]
NUMERICAS = [f for f in FEATURES if f not in CATEGORICAS and f not in NUMERICAS_LOG]

# Colunas que revelam o alvo ou servem apenas de identificação e validação.
PROIBIDAS = {"proficiencia", "alfabetizado", "alfabetizado_id", "presenca", "presenca_id", "preenchimento_caderno",
             "preenchimento_caderno_id", "peso_aluno", "peso_amostral", "taxa_alfabetizacao", "media_portugues",
             "gap_meta_ano", "gap_ate_2030", "atingiu_meta_ano", "atingiu_meta", "split", "ano", "ano_alvo",
             "id_aluno", "id_escola", "id_municipio", "nome_mesorregiao", "nome_municipio",
             *[f"proporcao_aluno_nivel_{i}" for i in range(9)]}

# Modelo por aluno (notebook 4): LightGBM escolhido na comparação.
ALVO_ALUNO = "alfabetizado"
PESO_ALUNO = "peso_amostral"
PESO_MAXIMO = 5
AMOSTRA_ALUNO = 50_000
AMOSTRA_ALUNO_TEMPORAL = 100_000
AMOSTRA_ALUNO_INTERPRETACAO = 5_000
BOOTSTRAP_ALUNO = 200
REPETICOES_PERMUTACAO_ALUNO = 3
FAMILIAS_ALUNO = {"rede": ["rede"], **{familia: colunas for familia, colunas in FAMILIAS.items() if familia != "metas"}, "escola": ["alunos_previstos_escola"]}
FEATURES_ALUNO = [f for colunas in FAMILIAS_ALUNO.values() for f in colunas]
CATEGORICAS_ALUNO = FAMILIAS_ALUNO["rede"] + CATEGORICAS
NUMERICAS_LOG_ALUNO = NUMERICAS_LOG + FAMILIAS_ALUNO["escola"]
NUMERICAS_ALUNO = [f for f in FEATURES_ALUNO if f not in CATEGORICAS_ALUNO and f not in NUMERICAS_LOG_ALUNO]
LIGHTGBM_ALUNO = {"n_estimators": 200, "learning_rate": .05, "num_leaves": 7, "min_child_samples": 1000, "subsample": .8, "subsample_freq": 1,
                  "colsample_bytree": .6, "random_state": SEED, "n_jobs": 1, "verbose": -1}

ARQ_BASE = DIR_DADOS / "base_municipio.parquet"
ARQ_BASE_2023 = DIR_DADOS / "base_municipio_2023.parquet"
ARQ_ESCORAGEM = DIR_DADOS / "base_municipio_escoragem_2025.parquet"
ARQ_PREPROCESSADOR = DIR_DADOS / "preprocessador_municipio.joblib"
ARQ_MODELO = DIR_DADOS / "modelo_taxa_municipal.joblib"
ARQ_MODELO_SEM_TERRITORIO = DIR_DADOS / "modelo_taxa_municipal_sem_territorio.joblib"
ARQ_RESIDUOS = DIR_DADOS / "residuos_oof_2024.parquet"
ARQ_PREVISOES = DIR_DADOS / "previsoes_2025.csv"
ARQ_CENARIOS = DIR_DADOS / "previsoes_2025_cenarios.csv"
ARQ_CATALOGO = DIR_DADOS / "catalogo_features.csv"
ARQ_EXCLUIDAS = DIR_DADOS / "features_excluidas.csv"
ARQ_BASE_ALUNO = DIR_DADOS / "base_aluno.parquet"
ARQ_PREPROCESSADOR_ALUNO = DIR_DADOS / "preprocessador_aluno.joblib"
ARQ_MODELO_ALUNO = DIR_DADOS / "modelo_aluno.joblib"
