"""Leitura das fontes dos modelos: camada Gold e Base dos Dados.

Conversão das seções de leitura de `2. Pre-processamento.ipynb`. Cada ano-alvo
recebe apenas dados de anos anteriores.
"""
import pandas as pd
from google.cloud import bigquery

from src.common import config

BD = config.BASE_DOS_DADOS
GOLD = config.GOLD
REF_INICIO, REF_FIM = min(config.ANOS_ALVO) - 1, max(config.ANOS_ALVO) - 1


def _ler_tabela(client: bigquery.Client, tabela: str) -> pd.DataFrame:
    return client.list_rows(tabela).to_dataframe()


def _consultar(client: bigquery.Client, sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()


def extrair(client: bigquery.Client) -> dict:
    """Retorna um DataFrame por fonte, sem transformações além das consultas."""
    redes = ", ".join(f'"{r}"' for r in config.REDES)
    consultas = {
        "participacao": f"""
            select ano, id_municipio, rede_id, count(*) as n_total, countif(presenca) as n_presentes,
                   countif(proficiencia is not null) as n_avaliados
            from `{GOLD}.fato_aluno`
            where rede_id in ({redes})
            group by 1, 2, 3""",
        "aluno": f"""
            select ano, id_escola, id_municipio, rede_id, alfabetizado_id as alfabetizado, peso_aluno, alunos_previstos_escola
            from (
              select *, count(*) over (partition by ano, id_escola) as alunos_previstos_escola
              from `{GOLD}.fato_aluno`
            )
            where proficiencia is not null and rede_id in ({redes})""",
        "atlas": f"""
            select id_municipio, idhm, idhm_e, renda_pc, indice_gini, prop_pobreza_criancas, expectativa_anos_estudo
            from `{BD}.mundo_onu_adh.municipio` where ano = {config.ANO_ATLAS}""",
        "populacao": f"""
            select ano + 1 as ano_alvo, ano as ano_ref, id_municipio, populacao
            from `{BD}.br_ibge_populacao.municipio` where ano between {REF_INICIO} and {REF_FIM}""",
        "pib": f"""
            select b.ano + 2 as ano_alvo, b.ano as ano_ref, b.id_municipio, safe_divide(b.pib, p.populacao) as pib_per_capita
            from `{BD}.br_ibge_pib.municipio` b
            join `{BD}.br_ibge_populacao.municipio` p on b.id_municipio = p.id_municipio and b.ano = p.ano
            where b.ano between {REF_INICIO - 1} and {REF_FIM - 1}""",
        "censo": f"""
            select ano + 1 as ano_alvo, ano as ano_ref, id_municipio, cast(safe_cast(rede as int64) as string) as rede_id,
                   count(*) as escolas_anos_iniciais,
                   sum(quantidade_matricula_fundamental_anos_iniciais) as matriculas_anos_iniciais,
                   100 * avg(if(safe_cast(tipo_localizacao as int64) = 2, 1, 0)) as pct_escolas_rurais,
                   100 * avg(safe_cast(biblioteca_sala_leitura as int64)) as pct_biblioteca,
                   100 * avg(safe_cast(esgoto_rede_publica as int64)) as pct_esgoto_rede,
                   safe_divide(sum(quantidade_matricula_fundamental_anos_iniciais), sum(quantidade_turma_fundamental_anos_iniciais)) as alunos_por_turma,
                   safe_divide(sum(quantidade_matricula_fundamental_anos_iniciais), sum(quantidade_docente_fundamental_anos_iniciais)) as alunos_por_docente
            from `{BD}.br_inep_censo_escolar.escola`
            where ano between {REF_INICIO} and {REF_FIM} and safe_cast(rede as int64) in (2, 3)
              and safe_cast(tipo_situacao_funcionamento as int64) = 1
              and safe_cast(etapa_ensino_fundamental_anos_iniciais as int64) = 1
            group by 1, 2, 3, 4""",
        "inep": f"""
            select id_municipio, if(lower(rede) = "estadual", "{config.REDE_ESTADUAL}", "{config.REDE_MUNICIPAL}") as rede_id,
                   safe_cast(atu_ef_2_ano as float64) as alunos_turma_2ano,
                   safe_cast(tdi_ef_anos_iniciais as float64) as distorcao_idade_serie,
                   safe_cast(taxa_aprovacao_ef_anos_iniciais as float64) as taxa_aprovacao,
                   safe_cast(taxa_abandono_ef_anos_iniciais as float64) as taxa_abandono,
                   safe_cast(dsu_ef_anos_iniciais as float64) as pct_docente_superior,
                   safe_cast(afd_ef_anos_iniciais_grupo_1 as float64) as pct_docente_adequado,
                   safe_cast(ird_baixa_regularidade as float64) as pct_docente_baixa_regularidade
            from `{BD}.br_inep_indicadores_educacionais.municipio`
            where ano = {config.ANO_INEP} and lower(localizacao) = "total" and lower(rede) in ("estadual", "municipal")""",
        "fundeb_municipal": f"""
            select ano + 1 as ano_alvo, ano as ano_ref, id_municipio, max(valor_real) as invest_aluno_fundamental
            from `{BD}.br_fnde_fundeb.indicador_municipal`
            where ano between {REF_INICIO} and {REF_FIM} and bimestre = 6 and cast(id_indicador as string) = "45"
            group by 1, 2, 3""",
        "fundeb_estadual": f"""
            select ano + 1 as ano_alvo, ano as ano_ref, sigla_uf, max(valor_real) as invest_aluno_fundamental
            from `{BD}.br_fnde_fundeb.indicador_estadual`
            where ano between {REF_INICIO} and {REF_FIM} and bimestre = 6 and cast(id_indicador as string) = "66"
            group by 1, 2, 3""",
        "territorio": f"""
            select id_municipio, nome_mesorregiao, cast(capital_uf as string) as capital_uf, cast(amazonia_legal as string) as amazonia_legal,
                   st_x(centroide) as lon, st_y(centroide) as lat
            from `{BD}.br_bd_diretorios_brasil.municipio`""",
    }
    tabelas = {"municipio": "dim_municipio", "rede": "dim_rede", "meta": "fato_meta", "indicador": "fato_indicador_municipio",
               "resultados": "fato_resultados"}
    fontes = {nome: _ler_tabela(client, f"{GOLD}.{tabela}") for nome, tabela in tabelas.items()}
    fontes.update({nome: _consultar(client, sql) for nome, sql in consultas.items()})
    return fontes
