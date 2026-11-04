# Tech Challenge Fase 3

Alunos:

* Artur Costa
* Davi Tolentino

Repositório do Tech Challenge da Fase 3 da pós-graduação em AI Scientist da FIAP.

Modelos de Machine Learning sobre o Indicador Criança Alfabetizada (2º ano do ensino fundamental, 2023 e 2024), construídos a partir da camada Gold da Fase 2 e enriquecidos com fontes públicas.

| Entregável | Onde está |
|---|---|
| Análise exploratória | [`notebooks/1. EDA.ipynb`](notebooks/1.%20EDA.ipynb) |
| Pipeline de dados e pré-processamento | [`notebooks/2. Pre-processamento.ipynb`](notebooks/2.%20Pre-processamento.ipynb) e [`src/preprocessing`](src/preprocessing) |
| Modelo de aluno (objetivo principal) | [`notebooks/4. Modelo de aluno.ipynb`](notebooks/4.%20Modelo%20de%20aluno.ipynb) |
| Modelo municipal (taxa e metas) | [`notebooks/3. Modelagem.ipynb`](notebooks/3.%20Modelagem.ipynb) |
| Perguntas de negócio | [`notebooks/5. Aplicacao Estrategica.ipynb`](notebooks/5.%20Aplicacao%20Estrategica.ipynb) |

## 1. Contexto do problema

A alfabetização ao fim do 2º ano é uma das bases do desenvolvimento educacional. Em 2024, cerca de 6 em cada 10 alunos avaliados das redes públicas foram considerados alfabetizados, com diferenças enormes entre estados: na rede municipal, a taxa vai de 36% na Bahia a 85% no Ceará. Os gestores precisam antecipar onde o risco está, entender quais fatores acompanham os resultados e saber quais municípios podem não atingir as metas do Compromisso Nacional Criança Alfabetizada.

## 2. Objetivo analítico

1. **Modelo de aluno:** prever se um aluno avaliado será alfabetizado, com variáveis educacionais, territoriais e socioeconômicas. É o objetivo principal do desafio.
2. **Modelo municipal:** prever a taxa de alfabetização da rede municipal e, a partir dela, a chance de cada município atingir a meta.
3. **Perguntas de negócio:** fatores associados, risco por município, metas futuras e variáveis mais influentes.

## 3. Base utilizada

| Fonte | Conteúdo | Uso |
|---|---|---|
| Camada Gold da Fase 2 (BigQuery, `tech-challenge-fase-2-505123.gold`) | Alunos (`fato_aluno`), indicadores, resultados, metas e dimensões | Alvos, histórico, rede, território e metas |
| Atlas do Desenvolvimento Humano (2010) | IDHM, renda, Gini, pobreza infantil | Socioeconômico |
| IBGE | População e PIB municipais | Socioeconômico |
| Censo Escolar (INEP) | Escolas, matrículas, ruralidade, infraestrutura, turmas e docentes | Infraestrutura e docentes |
| Indicadores educacionais (INEP, 2022) | Aprovação, abandono, distorção idade-série, formação docente | Fluxo escolar e docentes |
| Fundeb | Investimento por aluno | Financiamento |
| Diretórios da Base dos Dados | Mesorregião, capital, Amazônia Legal, coordenadas | Território e validação |

**População:** só os alunos avaliados das redes Estadual e Municipal (3.354.637 alunos, sendo 1.851.828 em 2024). Os não avaliados não têm alvo observado, e a taxa oficial também os exclui.

**Prevenção de vazamento:** todas as fontes externas e o histórico entram com ano anterior ao ano-alvo. `proficiencia`, presença, peso, IDs e médias do mesmo ano ficam fora das features; a lista completa, com justificativas, está em `data/processed/features_excluidas.csv`.

## 4. Etapas de modelagem

1. **EDA orientada a decisões:** cerca de 50 perguntas, cada uma ligada a uma decisão de modelagem.
2. **Engenharia de atributos:** 40 features em 11 famílias (histórico defasado, suavizado e de vizinhos; território; socioeconômico; infraestrutura; docentes; fluxo; financiamento; rede; escola; metas). São 38 no modelo de aluno e 38 no municipal.
3. **Pipeline scikit-learn:** `ColumnTransformer` com imputação pela mediana e indicador de ausência, `log1p` nas variáveis assimétricas, padronização e one-hot. O pré-processador é a primeira etapa do `Pipeline` e é ajustado só com o treino de cada dobra.
4. **Partições por mesorregião:** 70/15/15 de alunos em treino, validação e teste, sem nenhuma mesorregião em mais de uma partição e com todas as UFs no treino. Os dois modelos usam as mesmas partições.
5. **Seleção e otimização:** referências simples, KNN como benchmark e modelos lineares, árvores e boosting, com busca de hiperparâmetros em validação cruzada por mesorregião (`GroupKFold`).
6. **Generalização mais exigente:** estados nunca vistos, treino em 2023 e previsão de 2024, e bootstrap de mesorregiões para os intervalos de confiança.
7. **Interpretabilidade:** importância por permutação (por variável e por família) e SHAP.

## 5. Escolha do algoritmo

| Modelo | Algoritmo escolhido | Por quê |
|---|---|---|
| Aluno | LightGBM (200 árvores, `num_leaves=7`, `min_child_samples=1000`) | Empata com as logísticas na validação cruzada (AUC de 0,634 contra 0,638) e vai melhor na partição de validação (0,664) e no teste temporal (0,708 contra 0,697) |
| Município | Lasso (`alpha` de 0,0667) | Menor RMSE na validação cruzada (11,29) e menor sobreajuste; Ridge e LightGBM ficam dentro da variação entre dobras, e o empate favorece o modelo mais simples e interpretável |

## 6. Métricas de avaliação

O teste usa mesorregiões que não participaram de nenhum ajuste: 33 no modelo de aluno e 32 no municipal, que só tem a rede municipal.

**Modelo de aluno** (ROC-AUC ponderado pelo peso amostral; também PR-AUC, Brier, log loss e F1):

| Medida | Valor |
|---|---:|
| AUC no teste | 0,730 (IC 95% de 0,677 a 0,778) |
| Ganho sobre repetir a taxa de 2023 | +0,016 (IC de 0,006 a 0,028) |
| AUC em estados nunca vistos | 0,629 |
| AUC com treino em 2023, sem histórico | 0,708 |
| Acurácia balanceada com limiar de 0,63 | 0,669 (0,640 com 0,50) |

**Modelo municipal:**

| Medida | Valor |
|---|---:|
| RMSE / MAE / R² no teste | 11,27 / 8,35 / 0,71 |
| Ganho de RMSE sobre a persistência | 4,70 pontos (IC de 2,36 a 6,86) |
| AUC para atingir a meta de 2024 | 0,80 (0,76 sem território; 0,65 com treino em 2023) |

## 7. Interpretação dos resultados

- **O histórico do território domina os dois modelos.** `taxa_lag_suavizada` e `media_portugues_lag` lideram a permutação e o SHAP do modelo de aluno; embaralhar a família histórico derruba o AUC em 0,19.
- **O território vem em seguida.** No modelo municipal, retirar UF e região aumenta o RMSE em 1,54 ponto; a UF concentra 48,5% da variância da taxa municipal.
- **O contexto acrescenta pouco depois do histórico.** Cada família de contexto muda o AUC do aluno em no máximo 0,003 e o RMSE municipal em menos de 0,08.
- **O modelo de aluno descreve o contexto, não a criança.** Dentro de um município e rede, só `alunos_previstos_escola` varia, e conhecer a taxa real da escola levaria o AUC a apenas 0,80. Cerca de 85% da variância está dentro da escola.
- **As probabilidades do aluno são bem calibradas** (diferença máxima de 0,044 entre previsto e observado por decil), e agregadas por município reproduzem a taxa real com R² de 0,71.

## 8. Insights encontrados

1. **A desigualdade é sobretudo entre estados.** A UF concentra 48,5% da variância da taxa municipal, contra 13,1% da região, e renda e IDHM quase não diferenciam municípios de uma mesma UF. Isso aponta para diferenças de política e gestão estadual, que os dados não conseguem isolar.
2. **Fluxo escolar é o sinal de contexto mais forte.** Dentro da mesma UF, municípios no quartil superior de abandono têm taxas 10,5 pontos menores.
3. **341 municípios (6,4%) estão em risco alto** de ficar abaixo de 50% em 2025, somando cerca de 90 mil crianças avaliadas e não alfabetizadas previstas. Destacam-se SE (52% dos municípios) e BA (155 municípios). Manaus e Salvador concentram o maior volume.
4. **Cerca de 53% a 55% dos municípios devem atingir a meta de 2025.** 972 municípios estão em alerta nos dois cenários; 2.952 ficam incertos.
5. **O RS é um choque, não um padrão.** A queda de 2024 (-19 pontos) é generalizada no estado e muda muito a previsão de 2025 conforme o cenário.
6. **A ausência na prova não é aleatória** (de 1,9% no CE a 29,6% em SC), e escolas com mais ausentes têm taxas menores. A taxa pode estar superestimada onde a ausência é alta.

## 9. Limitações

- **Só uma transição entre anos (2023 → 2024).** Não há como validar a previsão de 2025 nem tendências até 2030.
- **Sem variáveis individuais.** O modelo de aluno tem teto baixo e não deve ser usado para avaliar crianças.
- **Associação, não causalidade.** Importâncias e SHAP não medem o efeito de intervenções.
- **Cobertura:** Roraima e DF ausentes em parte da Gold, sem microdados de SP e AC em 2023, rede 0 anômala na BA (excluída) e Atlas de 2010 defasado.
- **Informações indisponíveis na origem:** data de publicação das fontes externas, lista de municípios atingidos pelas enchentes no RS e momento em que a lista de alunos de cada escola (`alunos_previstos_escola`) é definida.
- **Municípios pequenos** têm taxas instáveis e erros maiores (MAE de 13,2 pontos com menos de 200 matrículas).

## 10. Aplicação prática para políticas públicas

- **Priorização de apoio técnico:** a lista de risco alto, com duas ordenações. Por taxa, para apoio intensivo em redes críticas; por número de crianças, para ganho de escala nas grandes redes.
- **Monitoramento de metas:** a probabilidade de atingir a meta de 2025 por município, com o alerta firme apenas quando os dois cenários concordam.
- **Foco no fluxo escolar:** abandono e distorção idade-série são sinais de alerta precoce, disponíveis antes da avaliação.
- **Cuidados de uso:** revisar municípios com poucos alunos avaliados e não usar as probabilidades como valores exatos.

## Estrutura do repositório

```
tech-challenge-fase-3/
├── data/processed/        bases analíticas, pré-processadores, modelos e previsões
├── notebooks/             1. EDA, 2. Pré-processamento, 3. Modelo municipal, 4. Modelo de aluno, 5. Aplicação estratégica
├── src/
│   ├── common/            configuração (caminhos, parâmetros e hiperparâmetros)
│   ├── preprocessing/     extração (BigQuery), features, bases, catálogo e pré-processadores
│   ├── modeling/          treino dos dois modelos e previsão de 2025
│   ├── evaluation/        métricas e avaliação no teste
│   └── run_pipeline.py    orquestrador
├── reports/avaliacao/     métricas, importâncias e previsões no teste dos dois modelos
├── images/                gráficos da EDA
├── docs/                  apresentação do projeto (HTML)
└── requirements.txt
```

## Como executar

Requisitos: Python 3.12 e as bibliotecas de `requirements.txt`.

```bash
pip install -r requirements.txt

# Pipeline completo: extração no BigQuery, treino e avaliação
python -m src.run_pipeline

# Sem acesso ao BigQuery, a partir das bases em data/processed
python -m src.run_pipeline --only modeling evaluation
```

A etapa `preprocessing` precisa de credenciais do Google Cloud com acesso ao projeto da Gold (`gcloud auth application-default login`). Os diretórios podem ser trocados com `DATA_DIR` e `REPORTS_DIR`, por exemplo para testar sem sobrescrever artefatos. O pipeline completo leva cerca de 10 minutos; os notebooks reproduzem as mesmas análises com mais detalhe.
