"""Orquestrador dos dois modelos: bases, treino, previsão e avaliação.

Execute a partir da raiz do repositório. A etapa `preprocessing` precisa de acesso
ao BigQuery; as demais usam apenas os arquivos já gerados.

Uso:
    python -m src.run_pipeline                                 # pipeline completo
    python -m src.run_pipeline --only modeling evaluation      # etapas específicas
"""
import argparse

ETAPAS = ["preprocessing", "modeling", "evaluation"]


def _executar(etapa: str) -> None:
    # Importa só o necessário: quem não usa o BigQuery não precisa da biblioteca instalada.
    if etapa == "preprocessing":
        from src.preprocessing import pipeline
        pipeline.run()
    elif etapa == "modeling":
        from src.modeling import prever, treinar, treinar_aluno
        treinar.run()
        prever.run()
        treinar_aluno.run()
    elif etapa == "evaluation":
        from src.evaluation import avaliar, avaliar_aluno
        avaliar.run()
        avaliar_aluno.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=ETAPAS, default=ETAPAS, help="Etapas a executar, sempre na ordem do pipeline.")
    selecionadas = parser.parse_args().only
    for etapa in [e for e in ETAPAS if e in selecionadas]:
        print(f"=== {etapa} ===")
        _executar(etapa)


if __name__ == "__main__":
    main()
