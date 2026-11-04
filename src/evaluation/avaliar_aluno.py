"""Avalia o modelo por aluno e grava as métricas em `reports/avaliacao/aluno`.

O ajuste usa treino e validação; o teste contém só mesorregiões que ficaram de fora.
Os testes por UF, temporal, de calibração e de segmentos seguem o notebook 4.

Uso:
    python -m src.evaluation.avaliar_aluno
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.common import config
from src.evaluation.metricas import bootstrap_grupos, variavel_original
from src.modeling import modelo as m
from src.modeling.treinar_aluno import ajustar, carregar_base, criar_modelo, prever


def metricas(y, prob, peso) -> dict:
    return {"auc": roc_auc_score(y, prob, sample_weight=peso), "pr_auc": average_precision_score(y, prob, sample_weight=peso),
            "brier": brier_score_loss(y, prob, sample_weight=peso), "log_loss": log_loss(y, prob, sample_weight=peso)}


def _alvo_peso(df: pd.DataFrame) -> tuple:
    return df[config.ALVO_ALUNO].values, df[config.PESO_ALUNO].values


def _validacao_cruzada(modelo, df: pd.DataFrame, grupo: str = config.GRUPO) -> dict:
    treino, teste = [], []
    for indices_treino, indices_teste in GroupKFold(n_splits=config.DOBRAS).split(df, groups=df[grupo]):
        ajustado = ajustar(modelo, df.iloc[indices_treino])
        treino.append(roc_auc_score(df.iloc[indices_treino][config.ALVO_ALUNO], prever(ajustado, df.iloc[indices_treino]), sample_weight=df.iloc[indices_treino][config.PESO_ALUNO]))
        teste.append(roc_auc_score(df.iloc[indices_teste][config.ALVO_ALUNO], prever(ajustado, df.iloc[indices_teste]), sample_weight=df.iloc[indices_teste][config.PESO_ALUNO]))
    return {"auc_treino": float(np.mean(treino)), "auc_cv": float(np.mean(teste)), "dp_cv": float(np.std(teste))}


def _escolher_limiar(y, prob, peso) -> float:
    """Limiar que maximiza a acurácia balanceada. Deve ser escolhido fora do teste."""
    limiares = np.arange(.3, .8, .01)
    return float(limiares[int(np.argmax([balanced_accuracy_score(y, prob >= limiar, sample_weight=peso) for limiar in limiares]))])


def _limiares(y, prob, peso, limiares: list) -> pd.DataFrame:
    linhas = []
    for limiar in limiares:
        classe = prob >= limiar
        linhas.append({"limiar": limiar, "acuracia_balanceada": balanced_accuracy_score(y, classe, sample_weight=peso), "f1_alfabetizado": f1_score(y, classe, sample_weight=peso),
                       "f1_nao_alfabetizado": f1_score(y, classe, pos_label=0, sample_weight=peso), "pct_previsto_alfabetizado": 100 * np.average(classe, weights=peso)})
    return pd.DataFrame(linhas)


def _calibracao(y, prob, peso) -> pd.DataFrame:
    df = pd.DataFrame({"prob": prob, "real": y, "peso": peso})
    faixas = pd.qcut(df["prob"], 10, duplicates="drop")
    linhas = [{"previsto": np.average(d["prob"], weights=d["peso"]), "observado": np.average(d["real"], weights=d["peso"]), "alunos": len(d)} for _, d in df.groupby(faixas, observed=True)]
    return pd.DataFrame(linhas).assign(cenario="LightGBM")


def _segmentos(teste: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    df = teste.assign(prob=prob, faixa=pd.cut(teste["matriculas_anos_iniciais"], config.FAIXAS_MATRICULAS, labels=config.ROTULOS_MATRICULAS, right=False))
    linhas = []
    for segmento, coluna in {"Região": "nome_regiao", "Rede": "rede", "Porte (matrículas)": "faixa", "UF": "sigla_uf"}.items():
        for grupo, d in df.groupby(coluna, observed=True):
            if d[config.ALVO_ALUNO].nunique() == 2 and len(d) >= 1000:
                y, peso = _alvo_peso(d)
                real, previsto = 100 * np.average(y, weights=peso), 100 * np.average(d["prob"], weights=peso)
                linhas.append({"segmento": segmento, "grupo": grupo, "alunos": len(d), "auc": roc_auc_score(y, d["prob"], sample_weight=peso), "real_pct": real, "previsto_pct": previsto, "vies": previsto - real})
    return pd.DataFrame(linhas)


def _teto(teste: pd.DataFrame, prob: np.ndarray) -> dict:
    y, peso = _alvo_peso(teste)
    grupos = {"municipio_rede": ["id_municipio", "rede_id"], "escola": ["id_escola"]}
    return {"modelo": roc_auc_score(y, prob, sample_weight=peso),
            **{nome: roc_auc_score(y, teste.groupby(chaves)[config.ALVO_ALUNO].transform("mean").values, sample_weight=peso) for nome, chaves in grupos.items()}}


def _permutacao(modelo, df: pd.DataFrame, grupos: dict) -> pd.DataFrame:
    """Queda do AUC ao embaralhar cada grupo de colunas, mantendo o modelo ajustado."""
    y, peso = _alvo_peso(df)
    base = roc_auc_score(y, prever(modelo, df), sample_weight=peso)
    rng = np.random.default_rng(config.SEED)
    linhas = []
    for nome, colunas in grupos.items():
        quedas = []
        for _ in range(config.REPETICOES_PERMUTACAO_ALUNO):
            permutado = df.copy()
            permutado[colunas] = permutado[colunas].values[rng.permutation(len(df))]
            quedas.append(base - roc_auc_score(y, prever(modelo, permutado), sample_weight=peso))
        linhas.append({"nome": nome, "queda_auc": np.mean(quedas), "dp": np.std(quedas)})
    return pd.DataFrame(linhas).sort_values("queda_auc", ascending=False)


def _shap(modelo, df: pd.DataFrame) -> pd.DataFrame:
    prep = modelo.named_steps["prep"]
    nomes = m.nomes_saida(prep)
    X = pd.DataFrame(prep.transform(df[config.FEATURES_ALUNO]), columns=nomes).astype(float)
    valores = shap.TreeExplainer(modelo.named_steps["modelo"])(X).values
    valores = pd.DataFrame(valores[..., -1] if valores.ndim == 3 else valores, columns=nomes)
    familias = {f: familia for familia, colunas in config.FAMILIAS_ALUNO.items() for f in colunas}
    df_shap = valores.T.groupby(variavel_original).sum().T.abs().mean().rename("shap_medio_abs").sort_values(ascending=False).rename_axis("variavel").reset_index()
    df_shap["familia"] = df_shap["variavel"].map(familias)
    return df_shap


def run() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    preprocessador = joblib.load(config.ARQ_PREPROCESSADOR_ALUNO)
    modelo = criar_modelo(preprocessador)
    df = carregar_base(colunas_extras=("id_escola", "id_municipio", "rede_id"))
    treino, validacao, teste = (df[df["split"] == s] for s in ("treino", "validacao", "teste"))
    ajuste = pd.concat([treino, validacao])
    assert not set(ajuste[config.GRUPO]) & set(teste[config.GRUPO]), "Mesorregião em ajuste e teste"
    y, peso = _alvo_peso(teste)
    grupos = teste[config.GRUPO].values
    amostra = treino.sample(config.AMOSTRA_ALUNO, random_state=config.SEED)

    # Limiar escolhido na validação, com um ajuste que não a usa.
    prob_validacao = prever(ajustar(modelo, amostra), validacao)
    y_validacao, peso_validacao = _alvo_peso(validacao)
    limiar = _escolher_limiar(y_validacao, prob_validacao, peso_validacao)

    ajustado = ajustar(modelo, ajuste)
    prob = prever(ajustado, teste)
    persistencia = teste["taxa_lag"].fillna(ajuste["taxa_lag"].median()).astype(float).values / 100
    proporcao = np.full(len(teste), np.average(ajuste[config.ALVO_ALUNO], weights=ajuste[config.PESO_ALUNO]))
    auc = lambda p, i: roc_auc_score(y[i], p[i], sample_weight=peso[i])
    ic_auc = bootstrap_grupos(grupos, lambda i: auc(prob, i), config.BOOTSTRAP_ALUNO)
    ic_ganho = bootstrap_grupos(grupos, lambda i: auc(prob, i) - auc(persistencia, i), config.BOOTSTRAP_ALUNO)

    df_uf = pd.concat([amostra, validacao.sample(config.AMOSTRA_ALUNO // 4, random_state=config.SEED)])
    cv = _validacao_cruzada(modelo, amostra)
    cv_uf = _validacao_cruzada(modelo, df_uf, "sigla_uf")
    sem_historico = criar_modelo(m.sem_colunas(preprocessador, config.FAMILIAS["historico"], True))
    cv_sem_historico = _validacao_cruzada(sem_historico, amostra)

    # Teste temporal: em 2023 todo o histórico é nulo, então a família sai do modelo.
    df_2023 = carregar_base(config.ANO_TREINO_TEMPORAL)
    temporal = metricas(y, prever(ajustar(sem_historico, df_2023.sample(min(config.AMOSTRA_ALUNO_TEMPORAL, len(df_2023)), random_state=config.SEED)), teste), peso)
    interpretacao = teste.sample(config.AMOSTRA_ALUNO_INTERPRETACAO * 5, random_state=config.SEED)

    resumo = {"modelo": {"algoritmo": "LightGBM", **{k: v for k, v in config.LIGHTGBM_ALUNO.items() if k not in ("n_jobs", "verbose")}, "features": len(config.FEATURES_ALUNO)},
              "bases": {"treino": len(treino), "validacao": len(validacao), "teste": len(teste), "mesorregioes_teste": int(teste[config.GRUPO].nunique()), "alunos_ajuste_final": len(df)},
              "validacao_cruzada": cv, "validacao_cruzada_uf": cv_uf, "validacao_cruzada_sem_historico": cv_sem_historico, "temporal": {**temporal, "alunos_2023": len(df_2023)},
              "ic95_auc": {"p2_5": ic_auc[0], "mediana": ic_auc[1], "p97_5": ic_auc[2]}, "ic95_ganho_auc_persistencia": {"p2_5": ic_ganho[0], "mediana": ic_ganho[1], "p97_5": ic_ganho[2]},
              "limiar_validacao": limiar, "teto_auc": _teto(teste, prob)}
    tabelas = {"desempenho_teste": pd.DataFrame([{"modelo": n, **metricas(y, p, peso)} for n, p in {"LightGBM": prob, "Persistência (taxa de 2023)": persistencia, "Proporção do ajuste": proporcao}.items()]),
               "limiares": _limiares(y, prob, peso, [.5, limiar]), "calibracao": _calibracao(y, prob, peso), "segmentos": _segmentos(teste, prob),
               "importancia_permutacao": _permutacao(ajustado, interpretacao, {f: [f] for f in config.FEATURES_ALUNO}),
               "importancia_familia": _permutacao(ajustado, interpretacao, config.FAMILIAS_ALUNO),
               "importancia_shap": _shap(ajustado, teste.sample(config.AMOSTRA_ALUNO_INTERPRETACAO, random_state=config.SEED))}

    saida = config.DIR_AVALIACAO / "aluno"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for nome, tabela in tabelas.items():
        tabela.to_csv(saida / f"{nome}.csv", index=False)
    print(tabelas["desempenho_teste"].round(3).to_string(index=False))
    print("Limiar:", round(limiar, 2), "| CV:", round(cv["auc_cv"], 3), "| UF:", round(cv_uf["auc_cv"], 3), "| Temporal:", round(temporal["auc"], 3))


if __name__ == "__main__":
    run()
