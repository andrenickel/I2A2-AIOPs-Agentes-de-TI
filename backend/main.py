#!/usr/bin/env python3
"""
Agente de IA com LangGraph para ingestão de Notas Fiscais (CSV dentro de um ZIP)
- Descompacta um ZIP contendo 2 arquivos CSV:
    * 202505_NFe_NotaFiscal
    * 202505_NFe_NotaFiscalItem
- Faz limpeza/normalização das colunas e tipos
- Valida headers esperados
- Carrega em PostgreSQL (tabelas: nfe_notafiscal, nfe_notafiscal_item)

Requisitos (pip):
    langgraph==0.2.*
    langchain-core>=0.3.0
    pandas>=2.1
    SQLAlchemy>=2.0
    psycopg2-binary>=2.9
    python-dotenv>=1.0
    pydantic>=2.7

Execução:
    export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/db"
    python main.py --zip caminho/para/arquivo.zip

Observações:
- O uso de LLM é opcional neste fluxo. Se desejar, defina OPENAI_API_KEY para ativar
  uma validação/explicação orientada por LLM (apenas log), sem bloquear a execução.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import zipfile
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from decimal import Decimal

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    create_engine,
    text,
    func,
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
from sqlalchemy.types import DECIMAL
from dotenv import load_dotenv

# --- API (opcional) ---
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

# ----------------------------------
# Config & Logging
# ----------------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("agent-nfe")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.warning("DATABASE_URL não definido. Defina a variável de ambiente antes de rodar.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # opcional
BATCH_SIZE = int(os.getenv("NFE_BATCH_SIZE", "1000"))  # tamanho do lote para upsert

# ----------------------------------
# Header Mappings (fornecidos pelo usuário)
# ----------------------------------
MAP_NFE = {
    "CHAVE DE ACESSO": "chave_acesso",
    "MODELO": "modelo",
    "SÉRIE": "serie",
    "NÚMERO": "numero",
    "NATUREZA DA OPERAÇÃO": "natureza_operacao",
    "DATA EMISSÃO": "data_emissao",
    "EVENTO MAIS RECENTE": "evento_mais_recente",
    "DATA/HORA EVENTO MAIS RECENTE": "datahora_evento_mais_recente",
    "CPF/CNPJ Emitente": "cpf_cnpj_emitente",
    "RAZÃO SOCIAL EMITENTE": "razao_social_emitente",
    "INSCRIÇÃO ESTADUAL EMITENTE": "inscricao_estadual_emitente",
    "UF EMITENTE": "uf_emitente",
    "MUNICÍPIO EMITENTE": "municipio_emitente",
    "CNPJ DESTINATÁRIO": "cnpj_destinatario",
    "NOME DESTINATÁRIO": "nome_destinatario",
    "UF DESTINATÁRIO": "uf_destinatario",
    "INDICADOR IE DESTINATÁRIO": "indicador_ie_destinatario",
    "DESTINO DA OPERAÇÃO": "destino_operacao",
    "CONSUMIDOR FINAL": "consumidor_final",
    "PRESENÇA DO COMPRADOR": "presenca_comprador",
    "VALOR NOTA FISCAL": "valor_nota_fiscal",
}

MAP_ITEM = {
    "CHAVE DE ACESSO": "chave_acesso",
    "NÚMERO PRODUTO": "numero_produto",
    "DESCRIÇÃO DO PRODUTO/SERVIÇO": "descricao_produto",
    "CÓDIGO NCM/SH": "codigo_ncm",
    "NCM/SH (TIPO DE PRODUTO)": "tipo_produto_ncm",
    "CFOP": "cfop",
    "QUANTIDADE": "quantidade",
    "UNIDADE": "unidade",
    "VALOR UNITÁRIO": "valor_unitario",
    "VALOR TOTAL": "valor_total",
}

# Colunas numéricas que podem vir com vírgula decimal
NUMERIC_COLS_NFE = ["valor_nota_fiscal"]
NUMERIC_COLS_ITEM = ["quantidade", "valor_unitario", "valor_total"]

DATE_COLS_NFE = ["data_emissao", "datahora_evento_mais_recente"]

# ----------------------------------
# SQLAlchemy Models
# ----------------------------------
Base = declarative_base()

class NotaFiscal(Base):
    __tablename__ = "nf_cabecalho"

    chave_acesso = Column(String(60), primary_key=True)
    modelo = Column(String(10))
    serie = Column(String(20))
    numero = Column(String(30))
    natureza_operacao = Column(String(255))
    data_emissao = Column(DateTime)
    evento_mais_recente = Column(String(120))
    datahora_evento_mais_recente = Column(DateTime)
    cpf_cnpj_emitente = Column(String(32), index=True)
    razao_social_emitente = Column(String(255))
    inscricao_estadual_emitente = Column(String(32))
    uf_emitente = Column(String(2))
    municipio_emitente = Column(String(120))
    cnpj_destinatario = Column(String(32), index=True)
    nome_destinatario = Column(String(255))
    uf_destinatario = Column(String(2))
    indicador_ie_destinatario = Column(String(10))
    destino_operacao = Column(String(20))
    consumidor_final = Column(String(10))
    presenca_comprador = Column(String(50))
    valor_nota_fiscal = Column(Numeric(18, 2))

class NotaFiscalItem(Base):
    __tablename__ = "nf_itens"

    # PK composta (chave_acesso + numero_produto)
    chave_acesso = Column(String(60), primary_key=True)
    numero_produto = Column(Integer, primary_key=True)

    descricao_produto = Column(String(500))
    codigo_ncm = Column(String(20), index=True)
    tipo_produto_ncm = Column(String(120))
    cfop = Column(String(10))
    quantidade = Column(Numeric(18, 4))
    unidade = Column(String(20))
    valor_unitario = Column(Numeric(18, 6))
    valor_total = Column(Numeric(18, 2))

class NotaFiscalAuditoria(Base):
    __tablename__ = "nf_auditoria"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chave_acesso = Column(String(60), index=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    checks = Column(JSONB, nullable=True)
    llm = Column(JSONB, nullable=True)

# ----------------------------------
# Utilitários de parsing & limpeza
# ----------------------------------
# ----------------------------------

def detect_encoding(sample: bytes) -> str:
    # estratégia simples: tenta utf-8, senão latin-1
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def normalize_decimal(s: Any) -> Optional[float]:
    if pd.isna(s):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    # remove separador de milhar e troca vírgula por ponto
    s = re.sub(r"\.(?=\d{3}(\D|$))", "", s)  # pontos como milhar
    s = s.replace(" ", "")
    s = s.replace("\u00A0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_datetime_pt(s: Any) -> Optional[datetime]:
    if pd.isna(s):
        return None
    s = str(s).strip()
    fmts = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def rename_and_cast(df: pd.DataFrame, mapping: Dict[str, str], numeric_cols: List[str], date_cols: Optional[List[str]] = None) -> pd.DataFrame:
    col_map = {old: mapping.get(old, old) for old in df.columns}
    df = df.rename(columns=col_map)
    # normalização básica de strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(normalize_decimal)
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].map(parse_datetime_pt)
    return df

def drop_null_keys(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    return df.dropna(subset=keys)

def dedupe_on_keys(df: pd.DataFrame, keys: List[str], label: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last")
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.warning("%s: removidos %s registros duplicados pelas chaves %s", label, removed, keys)
    return df

# ----------------------------------
# LangGraph: State & Nodes
# ----------------------------------
class IngestState(TypedDict, total=False):
    zip_path: str
    nfe_df: pd.DataFrame
    item_df: pd.DataFrame
    report: Dict[str, Any]
    errors: List[str]

# Minimal "LLM helper" (opcional): apenas para gerar mensagens de validação bonitinhas
class LLMValidator:
    def __init__(self, api_key: Optional[str] = None):
        self.enabled = bool(api_key)

    def explain_headers(self, found: List[str], expected_map: Dict[str, str], label: str) -> str:
        if not self.enabled:
            # Fallback simples
            missing = [h for h in expected_map.keys() if h not in found]
            extra = [h for h in found if h not in expected_map.keys()]
            return (
                f"[{label}] Headers OK? missing={missing or 'nenhum'}, extra={extra or 'nenhum'}"
            )
        # Se quisesse usar LLM aqui, seria o lugar. Mantemos simples para não exigir credenciais.
        missing = [h for h in expected_map.keys() if h not in found]
        extra = [h for h in found if h not in expected_map.keys()]
        return (
            f"[{label}] (LLM) Verificação de headers -> faltando: {missing or 'nenhum'}; "
            f"sobrando: {extra or 'nenhum'}"
        )

validator = LLMValidator(OPENAI_API_KEY)

# --- LLM (opcional) ---
HAVE_OPENAI = False
try:
    from openai import OpenAI  # novo SDK
    _client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    HAVE_OPENAI = _client is not None
except Exception:
    try:
        import openai  # legado
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            HAVE_OPENAI = True
    except Exception:
        HAVE_OPENAI = False


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def basic_audit_checks(nota: Dict[str, Any], itens: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues = []
    warnings = []
    soma_itens = sum(_to_float(i.get("valor_total")) or 0.0 for i in itens)
    valor_nf = _to_float(nota.get("valor_nota_fiscal")) or 0.0
    delta = abs(soma_itens - valor_nf)
    tol = max(0.01, 0.005 * max(valor_nf, soma_itens))  # 0.5% ou 0.01 mínimo
    if delta > tol:
        issues.append({
            "tipo": "DIVERGENCIA_TOTAL",
            "descricao": f"Soma dos itens ({soma_itens:.2f}) difere do valor da NF ({valor_nf:.2f}) além da tolerância ({tol:.2f}).",
            "sugestao": "Recalcular totais e verificar descontos/frete/seguro/encargos não mapeados no CSV."
        })
    for row in itens:
        ncm = str(row.get("codigo_ncm") or "").strip()
        if not ncm:
            warnings.append({"tipo":"NCM_VAZIO","descricao": f"Item {row.get('numero_produto')} sem NCM."})
        elif not (ncm.isdigit() and 4 <= len(ncm) <= 8):
            warnings.append({"tipo":"NCM_FORMATO","descricao": f"Item {row.get('numero_produto')} com NCM '{ncm}' suspeito."})
    for row in itens:
        q = _to_float(row.get("quantidade"))
        vu = _to_float(row.get("valor_unitario"))
        vt = _to_float(row.get("valor_total"))
        if q is None or vu is None or vt is None:
            continue
        if q <= 0 or vu < 0 or vt < 0:
            issues.append({
                "tipo": "VALORES_INVALIDOS",
                "descricao": f"Item {row.get('numero_produto')} com quantidade/valores não positivos.",
                "sugestao": "Verificar cadastro e origem do dado."
            })
        else:
            if abs(q*vu - vt) > max(0.01, 0.01*vt):
                warnings.append({
                    "tipo": "ARREDONDAMENTO",
                    "descricao": f"Item {row.get('numero_produto')} apresenta diferença entre q*vu e total (possível arredondamento/tributos embutidos)."
                })
    for row in itens:
        if not str(row.get("cfop") or "").strip():
            issues.append({"tipo":"CFOP_VAZIO","descricao": f"Item {row.get('numero_produto')} sem CFOP.", "sugestao":"Preencher CFOP conforme operação."})
    return {
        "soma_itens": round(soma_itens, 2),
        "valor_nf": round(valor_nf, 2),
        "delta": round(delta, 2),
        "tolerancia": round(tol, 2),
        "issues": issues,
        "warnings": warnings,
    }


def llm_audit(nota: Dict[str, Any], itens: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = (
        "Você é um auditor fiscal. Analise a Nota Fiscal eletrônica brasileira e seus itens."
        "Tarefas:"
        "- Verifique consistência com regras fiscais e possíveis divergências com cadastros."
        "- Identifique e sugira correções para: cálculo incorreto de impostos, CFOP/NCM incoerentes, divergências entre pedido de compra e NF."
        "- Produza um relatório objetivo com: Problemas encontrados, Riscos, Maiores agressores (itens ou campos que mais impactam), e Recomendações."
        "- Considere que nem todos os campos tributários podem estar no CSV; se faltar dado, indique quais faltam."
        f"Dados da NF (JSON):{json.dumps(nota, ensure_ascii=False, default=str)}"
        f"Itens (JSON):{json.dumps(itens, ensure_ascii=False, default=str)}"
        #"Responda somente o JSON com chaves: resumo, problemas[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}], riscos[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}], maiores_agressores[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}], recomendacoes[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}]."
        "Responda somente o JSON, sem marcações, com chaves: resumo, problemas[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}], riscos[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}], maiores_agressores[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}], recomendacoes[{\"campo\": \"campo\",\"descricao\": \"descricao.\"}]."
    )
    if not HAVE_OPENAI:
        return {
            "enabled": False,
            "message": "LLM não configurado (defina OPENAI_API_KEY). Retornando apenas checagens básicas.",
        }
    try:
        if 'OpenAI' in globals() and _client:
            resp = _client.chat.completions.create(
                model=os.getenv("NFE_LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
        else:
            resp = openai.ChatCompletion.create(
                model=os.getenv("NFE_LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role":"user","content": prompt}],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw": content}
        return {"enabled": True, "result": parsed}
    except Exception as e:
        return {"enabled": False, "error": str(e)}

# --- DB Engine ---
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# ----------------------------------
# Nós (funções) do fluxo
# ----------------------------------

def node_extract(state: IngestState) -> IngestState:
    zip_path = state.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        return {**state, "errors": ["ZIP não encontrado" ]}

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.info("Arquivos no ZIP: %s", names)

        # tenta achar os dois CSVs
        nfe_name = next((n for n in names if "NFe_NotaFiscal" in os.path.basename(n)), None)
        item_name = next((n for n in names if "NFe_NotaFiscalItem" in os.path.basename(n)), None)
        if not nfe_name or not item_name:
            return {**state, "errors": ["Arquivos esperados não encontrados no ZIP"]}

        # leitura com detecção simples de encoding
        def read_csv_from_zip(member: str) -> pd.DataFrame:
            with zf.open(member) as f:
                sample = f.read(2048)
                enc = detect_encoding(sample)
            with zf.open(member) as f2:
                # assume separador ; comum nos exports fiscais
                return pd.read_csv(f2, sep=";", encoding=enc, engine="python")

        nfe_df = read_csv_from_zip(nfe_name)
        item_df = read_csv_from_zip(item_name)

    logger.info("Lidos: nfe=%s linhas, item=%s linhas", len(nfe_df), len(item_df))

    return {**state, "nfe_df": nfe_df, "item_df": item_df}


def node_validate_clean(state: IngestState) -> IngestState:
    errors = state.get("errors", [])
    if errors:
        return state

    nfe_df = state["nfe_df"].copy()
    item_df = state["item_df"].copy()

    # logs bonitinhos (opcionalmente via LLM)
    msg_nfe = validator.explain_headers(list(nfe_df.columns), MAP_NFE, label="NotaFiscal")
    msg_item = validator.explain_headers(list(item_df.columns), MAP_ITEM, label="NotaFiscalItem")
    logger.info(msg_nfe)
    logger.info(msg_item)

    # renomeia e faz cast
    nfe_df = rename_and_cast(nfe_df, MAP_NFE, NUMERIC_COLS_NFE, DATE_COLS_NFE)
    item_df = rename_and_cast(item_df, MAP_ITEM, NUMERIC_COLS_ITEM, date_cols=None)

    # coerções finais das PKs e remoção de nulos nas chaves
    if "numero_produto" in item_df.columns:
        item_df["numero_produto"] = pd.to_numeric(item_df["numero_produto"], errors="coerce").astype("Int64")

    nfe_df = drop_null_keys(nfe_df, ["chave_acesso"]) if "chave_acesso" in nfe_df.columns else nfe_df
    item_df = drop_null_keys(item_df, ["chave_acesso", "numero_produto"]) if all(c in item_df.columns for c in ["chave_acesso", "numero_produto"]) else item_df

    # deduplicação para evitar CardinalityViolation no ON CONFLICT (lotes com chaves repetidas)
    nfe_df = dedupe_on_keys(nfe_df, ["chave_acesso"], label="NotaFiscal")
    if all(c in item_df.columns for c in ["chave_acesso", "numero_produto"]):
        item_df = dedupe_on_keys(item_df, ["chave_acesso", "numero_produto"], label="NotaFiscalItem")

    # sanity checks mínimos
    for must in ("chave_acesso",):
        if must not in nfe_df.columns:
            errors.append(f"Coluna obrigatória ausente em NotaFiscal: {must}")
        if must not in item_df.columns:
            errors.append(f"Coluna obrigatória ausente em NotaFiscalItem: {must}")

    if "numero_produto" not in item_df.columns:
        errors.append("Coluna obrigatória ausente em NotaFiscalItem: numero_produto")

    # casts finais
    if "numero_produto" in item_df.columns:
        item_df["numero_produto"] = pd.to_numeric(item_df["numero_produto"], errors="coerce").astype("Int64")

    return {**state, "nfe_df": nfe_df, "item_df": item_df, "errors": errors}


def ensure_schema(engine) -> None:
    Base.metadata.create_all(engine)


def upsert_dataframe(session: Session, df: pd.DataFrame, table, pkey_cols: List[str], batch_size: int = 1000) -> int:
    """Upsert em lotes para PostgreSQL usando INSERT ... ON CONFLICT DO UPDATE.
    Processa o DataFrame em chunks de até `batch_size` linhas e **garante** que
    não há chaves duplicadas dentro do mesmo comando (evita CardinalityViolation).
    """
    if df is None or df.empty:
        return 0
    # segurança extra: dedupe global
    df = df.drop_duplicates(subset=pkey_cols, keep="last")
    total = 0
    n = len(df)
    for start in range(0, n, batch_size):
        chunk = df.iloc[start:start + batch_size]
        # dedupe por lote também, por garantia
        chunk = chunk.drop_duplicates(subset=pkey_cols, keep="last")
        records = chunk.to_dict(orient="records")
        if not records:
            continue
        stmt = pg_insert(table).values(records)
        update_cols = {c.name: getattr(stmt.excluded, c.name) for c in table.__table__.columns if c.name not in pkey_cols}
        stmt = stmt.on_conflict_do_update(index_elements=pkey_cols, set_=update_cols)
        result = session.execute(stmt)
        total += (result.rowcount or len(records))
    return total


def node_load_db(state: IngestState) -> IngestState:
    errors = state.get("errors", [])
    if errors:
        return state

    if engine is None:
        errors.append("DATABASE_URL não configurado; impossível carregar no Postgres.")
        return {**state, "errors": errors}

    ensure_schema(engine)

    nfe_df = state["nfe_df"]
    item_df = state["item_df"]

    # remove colunas desconhecidas para evitar erro no upsert
    nfe_cols = {c.name for c in NotaFiscal.__table__.columns}
    item_cols = {c.name for c in NotaFiscalItem.__table__.columns}
    nfe_df = nfe_df[[c for c in nfe_df.columns if c in nfe_cols]]
    item_df = item_df[[c for c in item_df.columns if c in item_cols]]

    with Session(engine) as session:
        try:
            count_nfe = upsert_dataframe(session, nfe_df, NotaFiscal, ["chave_acesso"], batch_size=BATCH_SIZE)
            count_item = upsert_dataframe(session, item_df, NotaFiscalItem, ["chave_acesso", "numero_produto"], batch_size=BATCH_SIZE)
            session.commit()
            report = {
                "nfe_upserted": int(count_nfe),
                "item_upserted": int(count_item),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.info("Carga concluída: %s", report)
            return {**state, "report": report}
        except Exception as e:
            session.rollback()
            errors.append(f"Falha ao carregar no banco: {e}")
            logger.exception("Erro no load_db")
            return {**state, "errors": errors}


def node_finish(state: IngestState) -> IngestState:
    # apenas retorna estado final
    return state

# ----------------------------------
# Construção do grafo com LangGraph
# ----------------------------------
try:
    from langgraph.graph import StateGraph, END
except Exception as e:
    logger.error("LangGraph não instalado. pip install langgraph. Erro: %s", e)
    StateGraph = None
    END = None


def build_graph():
    if StateGraph is None:
        raise RuntimeError("LangGraph não está disponível. Instale as dependências.")

    workflow = StateGraph(IngestState)
    workflow.add_node("extract", node_extract)
    workflow.add_node("validate_clean", node_validate_clean)
    workflow.add_node("load_db", node_load_db)
    workflow.add_node("finish", node_finish)

    workflow.set_entry_point("extract")

    # Encadeamento simples
    workflow.add_edge("extract", "validate_clean")
    workflow.add_edge("validate_clean", "load_db")
    workflow.add_edge("load_db", "finish")
    workflow.add_edge("finish", END)

    return workflow.compile()

# ----------------------------------
# API (FastAPI)
# ----------------------------------
if 'HAVE_FASTAPI' in globals() and HAVE_FASTAPI:
    api = FastAPI(title="NFe Ingestion Agent", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from pydantic import BaseModel
    from typing import Optional, Any, Dict, List

    class IngestResponse(BaseModel):
        ok: bool
        report: Optional[Dict[str, Any]] = None
        errors: Optional[List[str]] = None
    class AnaliseRequest(BaseModel):
        chave_acesso: str

    class AnaliseResponse(BaseModel):
        ok: bool
        chave_acesso: str
        nota: Optional[Dict[str, Any]] = None
        itens_count: Optional[int] = None
        checks: Optional[Dict[str, Any]] = None
        llm: Optional[Dict[str, Any]] = None
        audit_id: Optional[int] = None
        errors: Optional[List[str]] = None

    def _serialize_nota(n: NotaFiscal) -> Dict[str, Any]:
        return {
            "chave_acesso": n.chave_acesso,
            "modelo": n.modelo,
            "serie": n.serie,
            "numero": n.numero,
            "natureza_operacao": n.natureza_operacao,
            "data_emissao": n.data_emissao,
            "evento_mais_recente": n.evento_mais_recente,
            "datahora_evento_mais_recente": n.datahora_evento_mais_recente,
            "cpf_cnpj_emitente": n.cpf_cnpj_emitente,
            "razao_social_emitente": n.razao_social_emitente,
            "inscricao_estadual_emitente": n.inscricao_estadual_emitente,
            "uf_emitente": n.uf_emitente,
            "municipio_emitente": n.municipio_emitente,
            "cnpj_destinatario": n.cnpj_destinatario,
            "nome_destinatario": n.nome_destinatario,
            "uf_destinatario": n.uf_destinatario,
            "indicador_ie_destinatario": n.indicador_ie_destinatario,
            "destino_operacao": n.destino_operacao,
            "consumidor_final": n.consumidor_final,
            "presenca_comprador": n.presenca_comprador,
            "valor_nota_fiscal": n.valor_nota_fiscal,
        }

    def _serialize_item(i: NotaFiscalItem) -> Dict[str, Any]:
        return {
            "chave_acesso": i.chave_acesso,
            "numero_produto": i.numero_produto,
            "descricao_produto": i.descricao_produto,
            "codigo_ncm": i.codigo_ncm,
            "tipo_produto_ncm": i.tipo_produto_ncm,
            "cfop": i.cfop,
            "quantidade": i.quantidade,
            "unidade": i.unidade,
            "valor_unitario": i.valor_unitario,
            "valor_total": i.valor_total,
        }

    @api.get("/health")
    def health():
        return {"status": "ok", "db": bool(engine is not None)}

    @api.post("/ingest", response_model=IngestResponse)
    async def ingest_zip(file: UploadFile = File(...)):
        if not file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Envie um arquivo .zip")
        if engine is None:
            raise HTTPException(status_code=500, detail="DATABASE_URL não configurado no servidor")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                tmp_path = tmp.name
        finally:
            await file.close()
        try:
            app_graph = build_graph()
            final_state = app_graph.invoke({"zip_path": tmp_path})
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if final_state.get("errors"):
            return IngestResponse(ok=False, errors=final_state["errors"])
        return IngestResponse(ok=True, report=final_state.get("report", {}))
    
    @api.get("/home")
    def home():
        """Retorna indicadores gerais:
        - valor_total (SUM de valor_nota_fiscal)
        - total_documentos (COUNT de nf_cabecalho)
        - documentos_analisados (COUNT de nf_auditoria)
        """
        if engine is None:
            raise HTTPException(status_code=500, detail="DATABASE_URL não configurado no servidor")
        
        with Session(engine) as session:
            valor_total = session.query(func.coalesce(func.sum(NotaFiscal.valor_nota_fiscal), 0)).scalar()
            total_documentos = session.query(func.count(NotaFiscal.chave_acesso)).scalar()
            documentos_analisados = session.query(func.count(NotaFiscalAuditoria.id)).scalar()
        # normaliza tipos numéricos para JSON
        def _to_number(x):
            try:
                return float(x)
            except Exception:
                try:
                    return int(x)
                except Exception:
                    return 0
        return {
            "valor_total": _to_number(valor_total),
            "total_documentos": int(total_documentos or 0),
            "documentos_analisados": int(documentos_analisados or 0),
        }

    @api.post("/analise")
    def analise(req: AnaliseRequest):
        if engine is None:
            return "DATABASE_URL não configurado no servidor"
        # 1) Cache: se já existe auditoria salva, retorna a mais recente
        try:
            with Session(engine) as session_cache:
                last = (
                    session_cache.query(NotaFiscalAuditoria)
                    .filter(NotaFiscalAuditoria.chave_acesso == req.chave_acesso)
                    .order_by(NotaFiscalAuditoria.created_at.desc(), NotaFiscalAuditoria.id.desc())
                    .first()
                )
                if last is not None:
                    return last.llm["result"]
        except Exception as e:
            logger.exception("Falha ao consultar cache de auditoria: %s", e)
        # 2) Busca e análise caso não tenha cache
        with Session(engine) as session:
            nota = session.get(NotaFiscal, req.chave_acesso)
            if nota is None:
                return AnaliseResponse(ok=False, chave_acesso=req.chave_acesso, errors=["Nota não encontrada"])
            itens = (
                session.query(NotaFiscalItem)
                .filter(NotaFiscalItem.chave_acesso == req.chave_acesso)
                .order_by(NotaFiscalItem.numero_produto.asc())
                .all()
            )
            nota_json = _serialize_nota(nota)
            itens_json = [_serialize_item(i) for i in itens]
            checks = basic_audit_checks(nota_json, itens_json)
            llm = llm_audit(nota_json, itens_json)
            # Persistência do relatório
            audit_id = None
            try:
                with Session(engine) as session2:
                    audit = NotaFiscalAuditoria(
                        chave_acesso=req.chave_acesso,
                        checks=checks,
                        llm=llm,
                    )
                    session2.add(audit)
                    session2.commit()
                    audit_id = int(audit.id)
            except Exception as e:
                logger.exception("Falha ao persistir auditoria: %s", e)
            return llm["result"]
        
    @api.get("/dashboard")
    def dashboard(inicio: Optional[str] = None, fim: Optional[str] = None, top_n: int = 4):
        """
        Endpoint para alimentar o dashboard fiscal.
        Parâmetros:
        - inicio, fim: YYYY-MM-DD (intervalo [inicio, fim))
        - top_n: número de fornecedores na pizza (demais viram "Outros")
        """
        if engine is None:
            raise HTTPException(status_code=500, detail="DATABASE_URL não configurado no servidor")

        from datetime import datetime, timedelta
        try:
            dt_fim = datetime.fromisoformat(fim) if fim else datetime.utcnow()
        except Exception:
            dt_fim = datetime.utcnow()
        try:
            dt_inicio = datetime.fromisoformat(inicio) if inicio else (dt_fim - timedelta(days=180))
        except Exception:
            dt_inicio = dt_fim - timedelta(days=180)

        with Session(engine) as session:
            # KPIs do período atual
            kpi_sql = text("""
                SELECT COALESCE(SUM(nc.valor_nota_fiscal),0) AS valor_total,
                    COUNT(nc.chave_acesso)               AS total_documentos,
                    (SELECT COUNT(*) FROM nf_auditoria na
                        WHERE na.created_at >= :inicio AND na.created_at < :fim) AS documentos_analisados,
                    (SELECT COUNT(*) FROM nf_auditoria na
                        WHERE na.created_at >= :inicio AND na.created_at < :fim
                            AND jsonb_array_length(COALESCE(na.checks->'issues','[]'::jsonb)) > 0) AS documentos_com_issues
                FROM nf_cabecalho nc
                WHERE nc.data_emissao >= :inicio AND nc.data_emissao < :fim
            """)
            row = session.execute(kpi_sql, {"inicio": dt_inicio, "fim": dt_fim}).mappings().first()
            valor_total = float(row["valor_total"]) if row and row["valor_total"] is not None else 0.0
            total_documentos = int(row["total_documentos"]) if row else 0
            documentos_analisados = int(row["documentos_analisados"]) if row else 0
            documentos_com_issues = int(row["documentos_com_issues"]) if row else 0

            taxa_erro = (documentos_com_issues / documentos_analisados) * 100.0 if documentos_analisados else 0.0
            processamento_automatico = (documentos_analisados / total_documentos) * 100.0 if total_documentos else 0.0

            # Período anterior (mesmo tamanho)
            delta = dt_fim - dt_inicio
            prev_inicio = dt_inicio - delta
            prev_fim = dt_inicio
            row_prev = session.execute(kpi_sql, {"inicio": prev_inicio, "fim": prev_fim}).mappings().first()
            valor_total_prev = float(row_prev["valor_total"]) if row_prev and row_prev["valor_total"] is not None else 0.0
            total_documentos_prev = int(row_prev["total_documentos"]) if row_prev else 0

            def pct_delta(cur, prev):
                return ((cur - prev) / prev * 100.0) if prev else None

            # Documentos por mês (usa .mappings() para vir dict-like)
            docs_mes_sql = text("""
                SELECT to_char(date_trunc('month', data_emissao), 'YYYY-MM') AS mes,
                    COUNT(*) AS total
                FROM nf_cabecalho
                WHERE data_emissao >= :inicio AND data_emissao < :fim
                GROUP BY 1
                ORDER BY 1
            """)
            documentos_por_mes = [
                {"mes": r["mes"], "total": int(r["total"])}
                for r in session.execute(docs_mes_sql, {"inicio": dt_inicio, "fim": dt_fim}).mappings().all()
            ]

            # Evolução do valor total por mês
            val_mes_sql = text("""
                select nc.data_emissao::timestamp::date mes,
                        sum(nc.valor_nota_fiscal) valor
                from nf_cabecalho nc 
                group by nc.data_emissao::timestamp::date
                order by 1 asc
            """)
            evolucao_valor_total = [
                {"mes": r["mes"], "valor": float(r["valor"]) if r["valor"] is not None else 0.0}
                for r in session.execute(val_mes_sql, {"inicio": dt_inicio, "fim": dt_fim}).mappings().all()
            ]

            # Distribuição por fornecedor
            forn_sql = text("""
                SELECT COALESCE(NULLIF(TRIM(razao_social_emitente),''), cpf_cnpj_emitente, 'Desconhecido') AS fornecedor,
                    COALESCE(SUM(valor_nota_fiscal),0) AS valor
                FROM nf_cabecalho
                WHERE data_emissao >= :inicio AND data_emissao < :fim
                GROUP BY 1
                ORDER BY valor DESC
                LIMIT :top_n
            """)
            top = [
                {"fornecedor": r["fornecedor"], "valor": float(r["valor"])}
                for r in session.execute(forn_sql, {"inicio": dt_inicio, "fim": dt_fim, "top_n": top_n}).mappings().all()
            ]
            top_total = sum(r["valor"] for r in top)
            others_val = max(0.0, valor_total - top_total)
            if others_val > 0:
                top.append({"fornecedor": "Outros", "valor": others_val})
            distribuicao_fornecedor = [
                {
                    "fornecedor": r["fornecedor"],
                    "valor": r["valor"],
                    "percentual": (r["valor"] / valor_total * 100.0) if valor_total else 0.0
                }
                for r in top
            ]

        return {
            "periodo": {
                "inicio": dt_inicio.isoformat(),
                "fim": dt_fim.isoformat(),
                "anterior": {"inicio": prev_inicio.isoformat(), "fim": prev_fim.isoformat()},
            },
            "kpis": {
                "total_documentos": total_documentos,
                "total_documentos_delta_pct": pct_delta(total_documentos, total_documentos_prev),
                "valor_total": valor_total,
                "valor_total_delta_pct": pct_delta(valor_total, valor_total_prev),
                "taxa_erro_pct": round(taxa_erro, 2),
                "processamento_automatico_pct": round(processamento_automatico, 2),
            },
            "documentos_por_mes": documentos_por_mes,
            "distribuicao_fornecedor": distribuicao_fornecedor,
            "evolucao_valor_total": evolucao_valor_total,
            "resumo_impostos": {
                "disponivel": False,
                "mensagem": "Campos tributários (ICMS/IPI/PIS/COFINS) não presentes no schema atual.",
                "icms": None, "ipi": None, "pis": None, "cofins": None
            }
        }


        
else:
    api = None

# ----------------------------------
# CLI
# ----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Agente de ingestão de NFe via LangGraph")
    parser.add_argument("--zip", dest="zip_path", required=False, help="Caminho para o arquivo ZIP")
    args = parser.parse_args(argv)

    app = build_graph()
    if not args.zip_path:
        print("Uso: python main.py --zip caminho/arquivo.zip  (ou rode a API: uvicorn main:api)")
        return 1

    app = build_graph()
    init_state: IngestState = {"zip_path": args.zip_path}
    final_state = app.invoke(init_state)

    if final_state.get("errors"):
        logger.error("Execução finalizada com erros: %s", final_state["errors"])
        print(json.dumps({"ok": False, "errors": final_state["errors"]}, ensure_ascii=False))
        return 2

    report = final_state.get("report", {})
    print(json.dumps({"ok": True, "report": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
