"""Avalia o modelo final e grava as métricas em `reports/avaliacao`.

O ajuste usa treino e validação; o teste contém só mesorregiões que ficaram de fora.
Os testes de meta, temporal e de calibração seguem os notebooks 3 e 4.

Uso:
    python -m src.evaluation.avaliar
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_validate

from src.common import config
from src.evaluation import metricas
from src.modeling import modelo as m


def _validacao_cruzada(modelo, treino: pd.DataFrame) -> dict:
    cv = cross_validate(modelo, treino[config.FEATURES], treino[config.ALVO].astype(float), groups=treino[config.GRUPO], cv=GroupKFold(n_splits=config.DOBRAS),
                        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}, return_train_score=True)
    return {"rmse_treino": -cv["train_rmse"].mean(), "rmse_cv": -cv["test_rmse"].mean(), "dp_cv": cv["test_rmse"].std(), "r2_cv": cv["test_r2"].mean()}


def _desempenho(teste: pd.DataFrame, ajuste: pd.DataFrame, previsto: np.ndarray, persistencia: np.ndarray) -> tuple:
    y = teste[config.ALVO].astype(float).values
    tabela = pd.DataFrame([{"modelo": "Lasso", **metricas.regressao(y, previsto)},
                           {"modelo": "Persistência (taxa de 2023)", **metricas.regressao(y, persistencia)},
                           {"modelo": "Média do ajuste", **metricas.regressao(y, np.full(len(y), ajuste[config.ALVO].astype(float).mean()))}])
    ic = metricas.bootstrap_grupos(teste[config.GRUPO].values, lambda i: root_mean_squared_error(y[i], persistencia[i]) - root_mean_squared_error(y[i], previsto[i]))
    return tabela, {"p2_5": ic[0], "mediana": ic[1], "p97_5": ic[2]}


def _segmentos(teste: pd.DataFrame, previsto: np.ndarray) -> tuple:
    df = teste.assign(erro=previsto - teste[config.ALVO].astype(float).values)
    df["erro_abs"], df["faixa"] = df["erro"].abs(), m.faixa_matriculas(df)
    linhas = []
    for segmento, coluna in {"Região": "nome_regiao", "Porte (matrículas)": "faixa"}.items():
        linhas.append(df.groupby(coluna).agg(municipios=("erro", "size"), mae=("erro_abs", "mean"), vies=("erro", "mean")).reset_index(names="grupo").assign(segmento=segmento))
    por_uf = df.groupby("sigla_uf").agg(municipios=("erro", "size"), mae=("erro_abs", "mean"), vies=("erro", "mean")).query("municipios >= 20").sort_values("mae", ascending=False).reset_index()
    return pd.concat(linhas), por_uf


def _metas(teste: pd.DataFrame, modelo_sem_territorio, previsto: np.ndarray, persistencia: np.ndarray) -> pd.DataFrame:
    y_meta, meta = teste[config.ALVO_META].values, teste["meta_ano"].astype(float).values
    placares = {"Lasso": previsto - meta, "Lasso sem território": modelo_sem_territorio.predict(teste[config.FEATURES]) - meta, "Persistência (taxa de 2023)": persistencia - meta}
    linhas = []
    for nome, placar in placares.items():
        ic = metricas.bootstrap_grupos(teste[config.GRUPO].values, lambda i: metricas.meta(y_meta[i], placar[i])["auc"], repeticoes=500)
        linhas.append({"abordagem": nome, **metricas.meta(y_meta, placar), "auc_min": ic[0], "auc_max": ic[2]})
    return pd.DataFrame(linhas)


def _temporal(df: pd.DataFrame, df_2023: pd.DataFrame, preprocessador) -> tuple:
    """Treina com o alvo de 2023 e prevê 2024: o teste mais próximo do uso real."""
    modelo = m.criar_modelo_sem_historico(preprocessador)
    residuos = m.residuos_oof(modelo, df_2023)
    modelo.fit(df_2023[config.FEATURES], df_2023[config.ALVO].astype(float))
    previsto = modelo.predict(df[config.FEATURES])
    y, meta, y_meta = df[config.ALVO].astype(float).values, df["meta_ano"].astype(float).values, df[config.ALVO_META].values
    resultado = {**metricas.regressao(y, previsto), **metricas.meta(y_meta, previsto - meta), "municipios_treino": len(df_2023), "municipios_previstos": len(df)}
    probabilidade = m.prob_acima(previsto, meta, m.faixa_matriculas(df), m.faixa_matriculas(df_2023), residuos)
    return resultado, probabilidade


def _calibracao(cenarios: dict) -> tuple:
    faixas, resumo = np.linspace(0, 1, 6), []
    curvas = []
    for nome, (probabilidade, real) in cenarios.items():
        curva = (pd.DataFrame({"prob": probabilidade, "real": real}).groupby(pd.cut(probabilidade, faixas, include_lowest=True), observed=True)
                 .agg(previsto=("prob", "mean"), observado=("real", "mean"), municipios=("real", "size")).reset_index(drop=True).assign(cenario=nome))
        curvas.append(curva)
        resumo.append({"cenario": nome, "brier": brier_score_loss(real, probabilidade), "brier_referencia": brier_score_loss(real, np.full(len(real), real.mean())),
                       "previsto_pct": 100 * probabilidade.mean(), "real_pct": 100 * real.mean()})
    return pd.concat(curvas), pd.DataFrame(resumo)


def _importancia(modelo, teste: pd.DataFrame) -> tuple:
    X, y = teste[config.FEATURES], teste[config.ALVO].astype(float)
    permutacao = permutation_importance(modelo, X, y, scoring="neg_root_mean_squared_error", n_repeats=config.REPETICOES_PERMUTACAO, random_state=config.SEED, n_jobs=1)
    df_permutacao = pd.DataFrame({"feature": config.FEATURES, "aumento_rmse": permutacao.importances_mean, "dp": permutacao.importances_std}).sort_values("aumento_rmse", ascending=False)
    prep = modelo.named_steps["prep"]
    nomes = m.nomes_saida(prep)
    X_prep = pd.DataFrame(prep.transform(X), columns=nomes).astype(float)
    masker = shap.maskers.Independent(X_prep.sample(min(config.AMOSTRA_SHAP, len(X_prep)), random_state=config.SEED), max_samples=config.AMOSTRA_SHAP)
    valores = pd.DataFrame(shap.Explainer(modelo.named_steps["modelo"], masker)(X_prep).values, columns=nomes)
    familias = {f: fam for fam, colunas in config.FAMILIAS.items() for f in colunas}
    df_shap = valores.T.groupby(metricas.variavel_original).sum().T.abs().mean().rename("shap_medio_abs").sort_values(ascending=False).rename_axis("variavel").reset_index()
    df_shap["familia"] = df_shap["variavel"].map(familias)
    return df_permutacao, df_shap


def run() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.read_parquet(config.ARQ_BASE)
    df_2023 = pd.read_parquet(config.ARQ_BASE_2023)
    preprocessador = joblib.load(config.ARQ_PREPROCESSADOR)
    producao = joblib.load(config.ARQ_MODELO)
    treino, ajuste, teste = df[df["split"] == "treino"], df[df["split"] != "teste"], df[df["split"] == "teste"]
    assert not set(ajuste[config.GRUPO]) & set(teste[config.GRUPO]), "Mesorregião em ajuste e teste"

    modelo = m.criar_modelo(preprocessador).fit(ajuste[config.FEATURES], ajuste[config.ALVO].astype(float))
    sem_territorio = m.criar_modelo_sem_territorio(preprocessador).fit(ajuste[config.FEATURES], ajuste[config.ALVO].astype(float))
    previsto = modelo.predict(teste[config.FEATURES])
    persistencia = teste["taxa_lag"].fillna(ajuste[config.ALVO].astype(float).median()).astype(float).values

    df_desempenho, ganho = _desempenho(teste, ajuste, previsto, persistencia)
    df_segmentos, df_uf = _segmentos(teste, previsto)
    df_metas = _metas(teste, sem_territorio, previsto, persistencia)
    temporal, prob_temporal = _temporal(df, df_2023, preprocessador)
    residuos_ajuste = m.residuos_oof(m.criar_modelo(preprocessador), ajuste)
    prob_teste = m.prob_acima(previsto, teste["meta_ano"].astype(float), m.faixa_matriculas(teste), m.faixa_matriculas(ajuste), residuos_ajuste)
    df_curvas, df_brier = _calibracao({"Mesorregiões novas, 2024": (prob_teste, teste[config.ALVO_META].values), "Treino 2023, previsão 2024": (prob_temporal, df[config.ALVO_META].values)})
    df_permutacao, df_shap = _importancia(modelo, teste)
    df_coeficientes = pd.Series(producao.named_steps["modelo"].coef_, index=m.nomes_saida(producao.named_steps["prep"]), name="coeficiente").rename_axis("termo").reset_index()
    df_coeficientes = df_coeficientes.sort_values("coeficiente", key=np.abs, ascending=False)

    resumo = {"validacao_cruzada": _validacao_cruzada(m.criar_modelo(preprocessador), treino), "ganho_rmse_sobre_persistencia": ganho, "temporal": temporal,
              "bases": {"treino": len(treino), "validacao": len(ajuste) - len(treino), "teste": len(teste), "mesorregioes_teste": int(teste[config.GRUPO].nunique())},
              "modelo": {"algoritmo": "Lasso", "alpha": config.LASSO_ALPHA, "features": len(config.FEATURES), "termos": len(df_coeficientes), "termos_ativos": int((df_coeficientes["coeficiente"] != 0).sum())}}
    config.DIR_AVALIACAO.mkdir(parents=True, exist_ok=True)
    (config.DIR_AVALIACAO / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    tabelas = {"desempenho_teste": df_desempenho, "erro_segmentos": df_segmentos, "erro_uf": df_uf, "metas_teste": df_metas, "calibracao_curvas": df_curvas, "calibracao_resumo": df_brier,
               "importancia_permutacao": df_permutacao, "importancia_shap": df_shap, "coeficientes": df_coeficientes,
               "previsoes_teste": pd.DataFrame({"id_municipio": teste["id_municipio"].values, "real": teste[config.ALVO].astype(float).values, "previsto": previsto})}
    for nome, tabela in tabelas.items():
        tabela.to_csv(config.DIR_AVALIACAO / f"{nome}.csv", index=False)
    print(df_desempenho.round(3).to_string(index=False))
    print(df_metas.round(3).to_string(index=False))
    print("Temporal:", {k: round(v, 3) for k, v in temporal.items()})


if __name__ == "__main__":
    run()
