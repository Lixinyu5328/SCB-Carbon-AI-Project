"""
Enrich DMA evidence for deduplicated representative IROs.

Purpose
-------
The initial secondary extraction has already produced one DMA evidence result
for every deduplicated IRO. This script leaves `unique` and `related` IROs
unchanged. For each `representative` IRO, it:

1. finds every member of the same merge group in the full merge mapping;
2. re-opens each member's original source document;
3. locates the member's original supporting excerpt and captures nearby context;
4. falls back to semantic top-k paragraph retrieval when the excerpt cannot be
   located reliably;
5. sends the representative's existing DMA result plus all member evidence
   bundles to one model call;
6. updates only applicable DMA fields when the combined evidence is more
   direct, specific, or complete;
7. retains exactly one final DMA result for the representative.

`unique` and `related` records are copied from the baseline JSONL without a new
model call. Merged members never receive separate DMA scores.

Default inputs
--------------
- sap_dma_secondary_extraction.jsonl
- sap_iro_dedup_register.csv
- sap_iro_merge_mapping.csv
- the original company-document corpus

Default outputs
---------------
- sap_dma_secondary_extraction_enriched.jsonl
  Complete final set: unchanged unique/related records plus enriched
  representative records.
- sap_dma_representative_enrichment_audit.csv
  Minimal audit trail for representative-only processing.
- sap_dma_representative_enrichment_failures.csv

Requires
--------
pip install openai pydantic pandas pdfplumber python-dotenv numpy
and OPENAI_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pdfplumber
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

from company_config import corpus_dir_for, get_company


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma

DEFAULT_BASELINE_JSONL = f"{COMPANY}_dma_secondary_extraction.jsonl"
DEFAULT_DEDUP_REGISTER = f"{COMPANY}_iro_dedup_register.csv"
DEFAULT_MERGE_MAPPING = f"{COMPANY}_iro_merge_mapping.csv"
DEFAULT_CORPUS_DIR = corpus_dir_for(COMPANY)

DEFAULT_JSONL_OUTPUT = f"{COMPANY}_dma_secondary_extraction_enriched.jsonl"
DEFAULT_CSV_OUTPUT = f"{COMPANY}_dma_secondary_extraction_enriched.csv"
DEFAULT_AUDIT_OUTPUT = f"{COMPANY}_dma_representative_enrichment_audit.csv"
DEFAULT_FAILURE_OUTPUT = f"{COMPANY}_dma_representative_enrichment_failures.csv"

DEFAULT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

TOP_K = 4
CONTEXT_PARAGRAPHS_BEFORE = 1
CONTEXT_PARAGRAPHS_AFTER = 1
MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 3000
MAX_MEMBER_CONTEXT_CHARS = 14_000
MAX_TOTAL_CONTEXT_CHARS = 120_000

RETRYABLE_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    ValueError,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

class IROType(str, Enum):
    negative_impact = "negative_impact"
    positive_impact = "positive_impact"
    risk = "risk"
    opportunity = "opportunity"


class CompanyIROStatus(str, Enum):
    actual = "actual"
    potential = "potential"
    unclear = "unclear"


class TimeHorizon(str, Enum):
    short_term = "short_term"
    medium_term = "medium_term"
    long_term = "long_term"


class FinancialEffectChannel(str, Enum):
    revenue_and_business_growth = "revenue_and_business_growth"
    costs_and_profitability = "costs_and_profitability"
    assets_liabilities_and_financial_position = (
        "assets_liabilities_and_financial_position"
    )
    cash_flow_and_liquidity = "cash_flow_and_liquidity"
    access_to_finance_and_cost_of_capital = (
        "access_to_finance_and_cost_of_capital"
    )
    business_continuity_and_strategic_viability = (
        "business_continuity_and_strategic_viability"
    )


class EvidenceSupport(str, Enum):
    explicit = "explicit"
    implied = "implied"
    not_disclosed = "not_disclosed"


class DimensionEvidence(BaseModel):
    support: EvidenceSupport
    evidence: str | None = None
    note: str | None = None


class FinancialChannelEvidence(BaseModel):
    channel: FinancialEffectChannel
    support: EvidenceSupport
    magnitude_evidence: str | None = None
    note: str | None = None


class DMAScoringEvidence(BaseModel):
    iro_id: str
    iro_type: IROType

    iro_status: CompanyIROStatus
    iro_status_evidence: str | None = None

    time_horizons: list[TimeHorizon] = Field(default_factory=list)
    time_horizon_support: EvidenceSupport
    time_horizon_evidence: str | None = None

    scale: DimensionEvidence | None = None
    scope: DimensionEvidence | None = None
    irremediability: DimensionEvidence | None = None

    financial_effects_status: EvidenceSupport | None = None
    financial_effects: list[FinancialChannelEvidence] = Field(default_factory=list)

    likelihood: DimensionEvidence


class FieldUpdateDecision(BaseModel):
    field_name: str
    action: str = Field(
        description=(
            "Use 'kept', 'supplemented' or 'replaced'. 'kept' is allowed "
            "only when the model explicitly reviewed a field but left it unchanged."
        )
    )
    reason: str = Field(
        description=(
            "A concise, evidence-grounded explanation of why the field was "
            "kept, supplemented or replaced."
        )
    )
    evidence_source_ids: list[str] = Field(
        default_factory=list,
        description=(
            "member_iro_id values of the evidence bundles used for this decision."
        ),
    )


class EnrichmentModelOutput(BaseModel):
    dma_result: DMAScoringEvidence
    field_updates: list[FieldUpdateDecision] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Enrichment prompt
# ---------------------------------------------------------------------------

ENRICHMENT_SYSTEM_PROMPT = """You are performing an ENRICHMENT PASS for one IRO that has already been identified, deduplicated and assessed in a prior secondary-extraction pass. You are given the representative IRO, its existing DMA evidence result, and evidence bundles for the representative plus every original IRO merged into it. All bundle members have already been judged to describe the same underlying IRO.

The complete secondary-extraction instructions below continue to apply without omission. In this enrichment pass, the supplied evidence consists of the representative evidence and all merged-member evidence bundles rather than one full company document.

You are an ESRS double-materiality evidence extraction expert.

You are performing a SECOND extraction pass for an IRO that has already been identified and deduplicated. Your task is to extract only the company-document evidence needed for later DMA scoring. Do not decide whether the IRO exists again, do not change its IRO type, and do not assign numerical scores.

Grounding rules
1. Use only the supplied company-document text and the existing IRO context.
2. Prefer short verbatim quotations. A close extract is allowed only when PDF text extraction has damaged formatting. Never invent facts, quantities, locations, affected groups, causal pathways, time horizons or financial effects.
3. Do not treat the number of document mentions as evidence of scope.
4. Do not use company size as evidence of scope unless the document links the specific IRO to that operational reach.
5. Exposure is not evidence of scale. The presence, diversity or vulnerability of affected groups does not show the intensity of harm or benefit unless the document describes the resulting impact.
6. General company reach is not evidence of scope. Total employee count, number of countries, supplier base, product category or intended product use may support scope only when the document links the specific impact to that reach.
7. For potential negative impacts, assess the gross impact: do not reduce the evidence because mitigation, controls or policies exist. Evidence about mitigation may be relevant to occurrence but must not be used to understate severity.
8. For risks and opportunities, extract gross potential financial effects before existing mitigation and controls.
9. For every applicable assessment dimension, return a structured support decision: explicit, implied or not_disclosed. Use not_disclosed when the company document does not provide usable evidence. Do not fill evidence merely because it seems plausible.
10. A policy, control, audit, grievance mechanism, commitment or remediation
process is not evidence of the scale of a positive impact unless the document
describes the resulting improvement and its significance.
11. The existence or size of an affected stakeholder group is not evidence
that an impact is actual, likely or widespread. Company-wide headcount or
reach supports scope only when the specific impact is linked to that group.
12. Industry-level impacts or risks are background evidence only. Do not use
them as evidence of company-specific scale, scope or likelihood unless the
document links the undertaking's own activities to the stated impact.
13. Do not use evidence from a different incident, country, site, affected
group or causal context to support the current IRO dimension unless the
document explicitly connects them.

Field rules by IRO type
A. positive_impact
- Extract iro_status, time horizon, scale, scope and likelihood.
- Return scale, scope and likelihood as DimensionEvidence objects, even when support is not_disclosed.
- Never return irremediability or financial effects.
- Scale means intensity/significance of the benefit to affected people or the environment.
- Scope means how widespread the benefit is across people, communities, locations, ecosystems or value-chain activities.

B. negative_impact
- Extract iro_status, time horizon, scale, scope, irremediability and likelihood.
- Return scale, scope, irremediability and likelihood as DimensionEvidence objects, even when support is not_disclosed.
- Never return financial effects.
- Scale means intensity/gravity of harm, not spread.
- Scope means spread across people, communities, locations, ecosystems or value-chain activities.
- Irremediability means whether restoration is possible and the time, cost and resources required.

C. risk or opportunity
- Extract iro_status, time horizon, likelihood and every financial-effect channel directly supported by the document.
- Return likelihood as a DimensionEvidence object.
- Set financial_effects_status to explicit or implied when one or more channels are supported; otherwise set it to not_disclosed and return an empty financial_effects list.
- Never return scale, scope or irremediability.
- Use only these mutually exclusive channels:
  1 revenue_and_business_growth: sales, demand, market share, market access or growth;
  2 costs_and_profitability: operating costs, capital expenditure, margins or profit;
  3 assets_liabilities_and_financial_position: asset values, impairments, provisions, liabilities or balance-sheet position;
  4 cash_flow_and_liquidity: amount, timing or stability of cash flows and ability to meet short-term obligations;
  5 access_to_finance_and_cost_of_capital: availability, terms or cost of debt, equity, insurance or other finance;
  6 business_continuity_and_strategic_viability: ability to maintain core operations or sustain a viable business model.
- Do not duplicate the same effect across channels. Choose the channel matching the immediate financial effect. Multiple channels are allowed only when each is separately supported.
- Each returned financial channel must include support, magnitude_evidence and an optional note. magnitude_evidence must describe qualitative significance or exposure through that channel. A generic sustainability statement is not financial magnitude evidence.
- Do not infer revenue, cost savings or another financial effect merely because a sustainability initiative, policy or product exists. The document must support the financial consequence itself; otherwise return financial_effects_status as not_disclosed and an empty list.
- Risk-management, due-diligence or sustainability measures do not bythemselves support a financial-effect channel or its likelihood. The document must support the specific financial consequence or occurrence pathway.

Status and likelihood
- For impacts, iro_status refers to the impact itself, not merely to the existence or implementation of a policy, programme, product, audit or action.
- actual: the impact itself has occurred or is currently occurring. For an actual impact, likelihood will later be fixed at 5, so likelihood.evidence must show current or recurring impact.
- potential: the impact, risk or opportunity may occur in the future or depends on a future pathway. An implemented action without evidence of its resulting impact normally supports potential rather than actual impact.
- unclear: the document does not allow a defensible actual/potential determination.
- likelihood.evidence should capture occurrence indicators such as historical events, repeated exposure, trends, approved plans, contractual commitments, scenario evidence or other direct support. Do not convert these into a 1-5 score.

Time horizons
- short_term, medium_term and long_term may all be selected if separately supported.
- Infer a horizon only from a clear date, plan period, asset life, target period or explicit near/medium/long-term statement. Otherwise return an empty list, set time_horizon_support to not_disclosed and leave time_horizon_evidence null.

Evidence support
- explicit: the document directly states the evidence.
- implied: the document provides specific evidence from which the dimension can reasonably be inferred.
- not_disclosed: the document does not provide usable evidence for that dimension.
- When support is not_disclosed, set the corresponding evidence field to null and briefly explain the missing disclosure in note where that field exists.

Return exactly one DMAScoringEvidence object. The iro_id and iro_type must exactly match the supplied values.

Additional rules for representative enrichment
1. Treat the existing DMA result as the baseline. Do not re-extract every field from zero.
2. Review every applicable DMA field using the representative evidence and all merged-member evidence together.
3. Keep the existing field unchanged when the new evidence does not provide a material improvement.
4. Replace or supplement an existing field only when the new evidence is more direct, more specific, more complete in coverage, or clearly stronger under the explicit/implied definitions.
5. A baseline not_disclosed field must remain not_disclosed unless the new evidence directly supports that exact DMA dimension. More information about the IRO generally is not sufficient to upgrade a specific DMA field.
6. Never downgrade explicit to implied or not_disclosed, and never downgrade implied to not_disclosed.
7. Do not remove an existing supported financial-effect channel merely because another member does not mention it.
8. Do not combine unrelated facts from different documents into a stronger claim. Evidence may be combined only when passages concern the same underlying IRO and are mutually compatible.
9. If separate sources provide complementary evidence for the same field, the final evidence may include more than one concise extract while preserving their source contexts.
10. When historical evidence shows that the impact itself has already occurred, classify the impact as actual unless the representative IRO specifically concerns a separate future pathway.
11. For positive impacts, apply separate tests for scale and scope:
   - Scale: the existence or implementation of a policy, programme, training, audit, grievance mechanism or remediation process does not support scale unless the document describes the resulting improvement and its significance for affected people or the environment.
   - Scope: the availability or operational coverage of such a policy, programme, training, audit, grievance mechanism or remediation process does not support scope unless the document links the resulting positive impact to the people, communities, locations, ecosystems or value-chain activities covered.
12. For negative impacts, numbers of employees, suppliers, countries, facilities or potentially vulnerable groups describe exposure only. They do not support scope unless the evidence links the specific adverse impact to that population, location or value-chain coverage.
13. Quantitative disparity, exposure or representation data may support the existence or scope of an IRO, but supports scale only when the evidence describes the seriousness or consequence of the resulting harm or benefit. Otherwise use implied or not_disclosed, not explicit.
14. Return exactly one final DMA result for the representative and never a separate result for a merged member.

Field-update reasoning and programmatic protection
- Return an EnrichmentModelOutput wrapper containing exactly one dma_result and a field_updates list.
- For every field whose final value differs from the supplied existing DMA result, include exactly one FieldUpdateDecision.
- field_name must be one of: iro_status, iro_status_evidence, time_horizons, time_horizon_support, time_horizon_evidence, scale, scope, irremediability, financial_effects_status, financial_effects, likelihood.
- action must be supplemented or replaced for a changed field.
- reason must identify the exact dimension-specific improvement and explain why it justifies the change under these rules.
- evidence_source_ids must contain the member_iro_id values of the bundles supporting the update.
- Do not report a changed field without a substantive reason and at least one valid evidence source id.
- You may omit unchanged fields from field_updates.
- The iro_id and iro_type inside dma_result must exactly match the representative values.
"""


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_quote_marks(text: str) -> str:
    return text.strip().strip("\"'“”‘’")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
    return records


def write_jsonl_atomic(path: str, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(target)


def load_completed_enriched_ids(path: str) -> set[str]:
    if not Path(path).exists():
        return set()
    return {str(r["iro_id"]) for r in read_jsonl(path)}


def append_jsonl(path: str, result: DMAScoringEvidence) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Document loading and chunking
# ---------------------------------------------------------------------------

class DocumentCache:
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self._text: dict[str, str] = {}
        self._chunks: dict[str, list[str]] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    def get_text(self, filename: str) -> str:
        if filename not in self._text:
            path = self.corpus_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Source document not found: {path}")
            self._text[filename] = self._extract(path)
        return self._text[filename]

    def get_chunks(self, filename: str) -> list[str]:
        if filename not in self._chunks:
            self._chunks[filename] = chunk_document(self.get_text(filename))
        return self._chunks[filename]

    def get_chunk_embeddings(self, filename: str) -> np.ndarray:
        if filename not in self._embeddings:
            chunks = self.get_chunks(filename)
            self._embeddings[filename] = embed_texts(chunks)
        return self._embeddings[filename]

    @staticmethod
    def _extract(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(pages).strip()
        elif suffix in {".txt", ".md"}:
            text = path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        else:
            raise ValueError(
                f"Unsupported source format {suffix!r} for {path.name}."
            )
        if not text:
            raise ValueError(
                f"No extractable text in {path.name}; OCR may be required."
            )
        return text


def chunk_document(
    text: str,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []

    for paragraph in raw:
        if len(paragraph) <= max_chars:
            if len(paragraph) >= min_chars:
                chunks.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars and current:
                if len(current) >= min_chars:
                    chunks.append(current)
                current = sentence
            else:
                current = candidate
        if len(current) >= min_chars:
            chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Excerpt location and semantic fallback
# ---------------------------------------------------------------------------

def excerpt_fragments(excerpt: str) -> list[str]:
    """
    Extract useful quote fragments. First-pass excerpts may contain several
    quotations separated by '/', semicolons, or narrative text.
    """
    excerpt = safe_text(excerpt)
    if not excerpt:
        return []

    quoted = re.findall(r'["“”](.+?)["“”]', excerpt, flags=re.DOTALL)
    candidates = quoted if quoted else re.split(r"\s*/\s*|\s*;\s*", excerpt)

    cleaned: list[str] = []
    for item in candidates:
        item = normalize_ws(strip_quote_marks(item))
        if len(item) >= 20 and item not in cleaned:
            cleaned.append(item)
    return cleaned or [normalize_ws(strip_quote_marks(excerpt))]


def find_chunk_indexes_for_excerpt(
    excerpt: str,
    chunks: list[str],
) -> tuple[list[int], str]:
    fragments = excerpt_fragments(excerpt)
    if not fragments or not chunks:
        return [], "not_found"

    normalized_chunks = [normalize_ws(c).lower() for c in chunks]

    exact_hits: list[int] = []
    for fragment in fragments:
        needle = normalize_ws(fragment).lower()
        for idx, chunk in enumerate(normalized_chunks):
            if needle and needle in chunk:
                exact_hits.append(idx)

    if exact_hits:
        return sorted(set(exact_hits)), "exact"

    # Conservative token-overlap fallback. It is used only to locate context,
    # not to classify evidence support.
    best_indexes: list[int] = []
    best_score = 0.0
    for fragment in fragments:
        tokens = {
            t for t in re.findall(r"[a-z0-9]+", fragment.lower())
            if len(t) >= 4
        }
        if not tokens:
            continue
        for idx, chunk in enumerate(normalized_chunks):
            chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk))
            score = len(tokens & chunk_tokens) / max(1, len(tokens))
            if score > best_score:
                best_score = score
                best_indexes = [idx]
            elif score == best_score and score > 0:
                best_indexes.append(idx)

    if best_score >= 0.72:
        return sorted(set(best_indexes[:3])), "fuzzy"

    return [], "not_found"


def expand_context(
    indexes: list[int],
    chunks: list[str],
    before: int = CONTEXT_PARAGRAPHS_BEFORE,
    after: int = CONTEXT_PARAGRAPHS_AFTER,
) -> list[str]:
    selected: list[int] = []
    for idx in indexes:
        start = max(0, idx - before)
        end = min(len(chunks), idx + after + 1)
        selected.extend(range(start, end))

    contexts: list[str] = []
    for idx in sorted(set(selected)):
        text = chunks[idx]
        if text not in contexts:
            contexts.append(text)
    return contexts


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0))
    vectors: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=float)


def cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if query.size == 0 or matrix.size == 0:
        return np.asarray([])
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    matrix_norm = matrix / (
        np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    )
    return matrix_norm @ query_norm


def build_retrieval_query(
    representative: pd.Series,
    member: pd.Series,
) -> str:
    fields = [
        safe_text(representative.get("esrs_topic")),
        safe_text(representative.get("esrs_subtopic")),
        safe_text(representative.get("esrs_sub_subtopic")),
        safe_text(representative.get("iro_type")),
        safe_text(member.get("supporting_excerpt")),
        safe_text(member.get("reasoning")),
    ]
    return normalize_ws(" ".join(value for value in fields if value))


def retrieve_top_k(
    filename: str,
    query: str,
    doc_cache: DocumentCache,
    k: int = TOP_K,
) -> list[dict[str, Any]]:
    chunks = doc_cache.get_chunks(filename)
    if not chunks or not query:
        return []

    query_embedding = embed_texts([query])[0]
    chunk_embeddings = doc_cache.get_chunk_embeddings(filename)
    similarities = cosine_similarities(query_embedding, chunk_embeddings)
    top_indexes = np.argsort(similarities)[::-1][:k]

    return [
        {
            "similarity": round(float(similarities[idx]), 4),
            "text": chunks[int(idx)],
        }
        for idx in top_indexes
    ]


def truncate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(bundle, ensure_ascii=False)
    if len(encoded) <= MAX_MEMBER_CONTEXT_CHARS:
        return bundle

    shortened = dict(bundle)
    for key in ("located_contexts", "retrieved_top_k"):
        items = shortened.get(key, [])
        while items and len(json.dumps(shortened, ensure_ascii=False)) > (
            MAX_MEMBER_CONTEXT_CHARS
        ):
            items.pop()
        shortened[key] = items
    return shortened


def build_member_bundle(
    representative: pd.Series,
    member: pd.Series,
    doc_cache: DocumentCache,
) -> dict[str, Any]:
    filename = safe_text(member.get("filename"))
    chunks = doc_cache.get_chunks(filename)
    excerpt = safe_text(member.get("supporting_excerpt"))

    indexes, location_mode = find_chunk_indexes_for_excerpt(excerpt, chunks)
    located_contexts = expand_context(indexes, chunks) if indexes else []
    retrieved_top_k: list[dict[str, Any]] = []

    if not located_contexts:
        query = build_retrieval_query(representative, member)
        retrieved_top_k = retrieve_top_k(
            filename=filename,
            query=query,
            doc_cache=doc_cache,
            k=TOP_K,
        )
        location_mode = "semantic_top_k"

    bundle = {
        "member_iro_id": safe_text(member.get("iro_uid")),
        "dedup_role": safe_text(member.get("dedup_role")),
        "source_filename": filename,
        "rule_id": safe_text(member.get("rule_id")),
        "esrs_topic": safe_text(member.get("esrs_topic")),
        "esrs_subtopic": safe_text(member.get("esrs_subtopic")),
        "esrs_sub_subtopic": safe_text(member.get("esrs_sub_subtopic")),
        "original_evidence_status": safe_text(member.get("evidence_status")),
        "original_supporting_excerpt": excerpt,
        "original_reasoning": safe_text(member.get("reasoning")),
        "location_mode": location_mode,
        "located_contexts": located_contexts,
        "retrieved_top_k": retrieved_top_k,
    }
    return truncate_bundle(bundle)


# ---------------------------------------------------------------------------
# Input validation and merge-group preparation
# ---------------------------------------------------------------------------

REQUIRED_REGISTER_COLUMNS = {
    "iro_uid",
    "dedup_role",
    "iro_type",
    "filename",
}
REQUIRED_MAPPING_COLUMNS = {
    "iro_uid",
    "merge_group_id",
    "dedup_role",
    "iro_type",
    "filename",
    "supporting_excerpt",
    "reasoning",
}


def validate_inputs(
    register: pd.DataFrame,
    mapping: pd.DataFrame,
    baseline_records: list[dict[str, Any]],
) -> None:
    missing_register = REQUIRED_REGISTER_COLUMNS - set(register.columns)
    missing_mapping = REQUIRED_MAPPING_COLUMNS - set(mapping.columns)
    if missing_register:
        raise ValueError(
            f"Dedup register missing columns: {sorted(missing_register)}"
        )
    if missing_mapping:
        raise ValueError(
            f"Merge mapping missing columns: {sorted(missing_mapping)}"
        )

    valid_roles = {"unique", "related", "representative"}
    unexpected = set(register["dedup_role"].dropna().astype(str)) - valid_roles
    if unexpected:
        raise ValueError(
            f"Unexpected dedup_role values in register: {sorted(unexpected)}"
        )

    baseline_ids = {str(r["iro_id"]) for r in baseline_records}
    register_ids = set(register["iro_uid"].astype(str))
    missing_baseline = register_ids - baseline_ids
    if missing_baseline:
        raise ValueError(
            "Baseline JSONL is missing deduplicated IRO result(s): "
            f"{sorted(missing_baseline)[:10]}"
        )

    representatives = register[register["dedup_role"] == "representative"]
    for _, rep in representatives.iterrows():
        group_id = safe_text(rep.get("merge_group_id"))
        if not group_id:
            raise ValueError(
                f"Representative {rep['iro_uid']} has no merge_group_id."
            )
        members = mapping[
            (mapping["merge_group_id"].astype(str) == group_id)
            & mapping["dedup_role"].isin(["representative", "merged"])
        ]
        if len(members) < 2:
            raise ValueError(
                f"Merge group {group_id} has fewer than two active members."
            )
        rep_count = (
            members["dedup_role"].astype(str) == "representative"
        ).sum()
        if rep_count != 1:
            raise ValueError(
                f"Merge group {group_id} has {rep_count} representatives."
            )
        if len(set(members["iro_type"].astype(str))) != 1:
            raise ValueError(
                f"Merge group {group_id} contains multiple iro_type values."
            )


def get_group_members(
    representative: pd.Series,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    group_id = safe_text(representative.get("merge_group_id"))
    members = mapping[
        (mapping["merge_group_id"].astype(str) == group_id)
        & mapping["dedup_role"].isin(["representative", "merged"])
    ].copy()

    # Representative first, then merged members in stable input order.
    role_order = {"representative": 0, "merged": 1}
    members["_role_order"] = (
        members["dedup_role"].map(role_order).fillna(2)
    )
    return (
        members.sort_values(["_role_order", "iro_uid"])
        .drop(columns=["_role_order"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Prompt construction and model call
# ---------------------------------------------------------------------------

def build_enrichment_prompt(
    representative: pd.Series,
    baseline: dict[str, Any],
    bundles: list[dict[str, Any]],
) -> str:
    payload = {
        "representative_context": {
            "iro_id": safe_text(representative.get("iro_uid")),
            "merge_group_id": safe_text(
                representative.get("merge_group_id")
            ),
            "iro_type": safe_text(representative.get("iro_type")),
            "rule_id": safe_text(representative.get("rule_id")),
            "standard": safe_text(representative.get("standard")),
            "esrs_topic": safe_text(representative.get("esrs_topic")),
            "esrs_subtopic": safe_text(
                representative.get("esrs_subtopic")
            ),
            "esrs_sub_subtopic": safe_text(
                representative.get("esrs_sub_subtopic")
            ),
            "dedup_rationale": safe_text(
                representative.get("dedup_rationale")
            ),
        },
        "existing_dma_result": baseline,
        "all_merge_group_evidence_bundles": bundles,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) > MAX_TOTAL_CONTEXT_CHARS:
        raise ValueError(
            f"Combined context for {representative['iro_uid']} is "
            f"{len(text):,} characters, above "
            f"MAX_TOTAL_CONTEXT_CHARS={MAX_TOTAL_CONTEXT_CHARS:,}. "
            "Reduce TOP_K or context limits."
        )

    return (
        "Compare the existing result with all evidence bundles and return one "
        "final result for the representative:\n\n" + text
    )


def make_not_disclosed_dimension(note: str) -> DimensionEvidence:
    return DimensionEvidence(
        support=EvidenceSupport.not_disclosed,
        evidence=None,
        note=note,
    )


def normalize_dimension(
    dimension: DimensionEvidence | None,
) -> DimensionEvidence | None:
    if dimension is None:
        return None
    if dimension.support == EvidenceSupport.not_disclosed:
        dimension.evidence = None
    return dimension


def reconcile_gaps(result: DMAScoringEvidence) -> DMAScoringEvidence:
    result.scale = normalize_dimension(result.scale)
    result.scope = normalize_dimension(result.scope)
    result.irremediability = normalize_dimension(result.irremediability)
    result.likelihood = normalize_dimension(result.likelihood)

    if result.likelihood is None:
        result.likelihood = make_not_disclosed_dimension(
            "No usable likelihood evidence was disclosed."
        )

    if not result.time_horizons:
        result.time_horizon_support = EvidenceSupport.not_disclosed
        result.time_horizon_evidence = None
    elif result.time_horizon_support == EvidenceSupport.not_disclosed:
        result.time_horizons = []
        result.time_horizon_evidence = None

    if result.iro_type == IROType.positive_impact:
        result.scale = result.scale or make_not_disclosed_dimension(
            "No usable scale evidence was disclosed."
        )
        result.scope = result.scope or make_not_disclosed_dimension(
            "No usable scope evidence was disclosed."
        )
        result.irremediability = None
        result.financial_effects_status = None
        result.financial_effects = []

    elif result.iro_type == IROType.negative_impact:
        result.scale = result.scale or make_not_disclosed_dimension(
            "No usable scale evidence was disclosed."
        )
        result.scope = result.scope or make_not_disclosed_dimension(
            "No usable scope evidence was disclosed."
        )
        result.irremediability = (
            result.irremediability
            or make_not_disclosed_dimension(
                "No usable irremediability evidence was disclosed."
            )
        )
        result.financial_effects_status = None
        result.financial_effects = []

    else:
        result.scale = None
        result.scope = None
        result.irremediability = None

        effects: list[FinancialChannelEvidence] = []
        seen: set[FinancialEffectChannel] = set()
        for effect in result.financial_effects:
            if effect.channel in seen:
                continue
            if effect.support == EvidenceSupport.not_disclosed:
                continue
            if not safe_text(effect.magnitude_evidence):
                continue
            effects.append(effect)
            seen.add(effect.channel)

        result.financial_effects = effects
        if not effects:
            result.financial_effects_status = EvidenceSupport.not_disclosed
        elif any(e.support == EvidenceSupport.explicit for e in effects):
            result.financial_effects_status = EvidenceSupport.explicit
        else:
            result.financial_effects_status = EvidenceSupport.implied

    return result


SUPPORT_RANK = {
    EvidenceSupport.not_disclosed.value: 0,
    EvidenceSupport.implied.value: 1,
    EvidenceSupport.explicit.value: 2,
}


def guard_against_support_downgrade(
    baseline: dict[str, Any],
    enriched: DMAScoringEvidence,
) -> DMAScoringEvidence:
    """
    Deterministic safety guard: the enrichment pass may add or strengthen
    evidence, but it may not lower a structured support level or silently
    remove a previously supported financial channel.
    """
    final = enriched.model_dump(mode="json")

    for field in ("scale", "scope", "irremediability", "likelihood"):
        old = baseline.get(field)
        new = final.get(field)
        if not old or not new:
            continue
        old_rank = SUPPORT_RANK.get(str(old.get("support")), -1)
        new_rank = SUPPORT_RANK.get(str(new.get("support")), -1)
        if new_rank < old_rank:
            final[field] = old

    old_time_rank = SUPPORT_RANK.get(
        str(baseline.get("time_horizon_support")), -1
    )
    new_time_rank = SUPPORT_RANK.get(
        str(final.get("time_horizon_support")), -1
    )
    if new_time_rank < old_time_rank:
        final["time_horizons"] = baseline.get("time_horizons", [])
        final["time_horizon_support"] = baseline.get(
            "time_horizon_support"
        )
        final["time_horizon_evidence"] = baseline.get(
            "time_horizon_evidence"
        )

    old_effects = {
        item["channel"]: item
        for item in baseline.get("financial_effects", [])
        if item.get("channel")
    }
    new_effects = {
        item["channel"]: item
        for item in final.get("financial_effects", [])
        if item.get("channel")
    }

    for channel, old_item in old_effects.items():
        new_item = new_effects.get(channel)
        if new_item is None:
            new_effects[channel] = old_item
            continue
        old_rank = SUPPORT_RANK.get(str(old_item.get("support")), -1)
        new_rank = SUPPORT_RANK.get(str(new_item.get("support")), -1)
        if new_rank < old_rank:
            new_effects[channel] = old_item

    final["financial_effects"] = list(new_effects.values())

    # Recompute financial_effects_status through reconcile_gaps.
    return reconcile_gaps(DMAScoringEvidence.model_validate(final))


def enrich_representative(
    representative: pd.Series,
    baseline: dict[str, Any],
    bundles: list[dict[str, Any]],
    model: str,
) -> tuple[DMAScoringEvidence, list[FieldUpdateDecision]]:
    prompt = build_enrichment_prompt(representative, baseline, bundles)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": ENRICHMENT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=EnrichmentModelOutput,
            )
            if response.output_parsed is None:
                raise ValueError(
                    "Model returned no parsed output (possible refusal)."
                )

            parsed = response.output_parsed
            result = reconcile_gaps(parsed.dma_result)
            expected_id = safe_text(representative.get("iro_uid"))
            expected_type = safe_text(representative.get("iro_type"))

            if result.iro_id != expected_id:
                raise ValueError(
                    f"Model changed iro_id from {expected_id!r} to "
                    f"{result.iro_id!r}."
                )
            if result.iro_type.value != expected_type:
                raise ValueError(
                    f"Model changed iro_type from {expected_type!r} to "
                    f"{result.iro_type.value!r}."
                )

            guarded_result = guard_against_support_downgrade(
                baseline, result
            )
            guarded_record = guarded_result.model_dump(mode="json")
            final_changes = changed_fields(baseline, guarded_record)

            allowed_fields = set(AUDITED_FIELDS)
            valid_source_ids = {
                safe_text(bundle.get("member_iro_id"))
                for bundle in bundles
            }
            decisions_by_field: dict[str, FieldUpdateDecision] = {}
            for decision in parsed.field_updates:
                if decision.field_name not in allowed_fields:
                    raise ValueError(
                        f"Unsupported field_update field_name: "
                        f"{decision.field_name!r}."
                    )
                if decision.field_name in decisions_by_field:
                    raise ValueError(
                        f"Duplicate field_update decision for "
                        f"{decision.field_name!r}."
                    )
                if decision.action not in {"supplemented", "replaced", "kept"}:
                    raise ValueError(
                        f"Unsupported field_update action: "
                        f"{decision.action!r}."
                    )
                invalid_ids = set(decision.evidence_source_ids) - valid_source_ids
                if invalid_ids:
                    raise ValueError(
                        "field_update contains unknown evidence_source_ids: "
                        f"{sorted(invalid_ids)}"
                    )
                decisions_by_field[decision.field_name] = decision

            missing_reasons = [
                field for field in final_changes
                if field not in decisions_by_field
                or decisions_by_field[field].action not in {
                    "supplemented", "replaced"
                }
                or not decisions_by_field[field].reason.strip()
                or not decisions_by_field[field].evidence_source_ids
            ]
            if missing_reasons:
                raise ValueError(
                    "Changed field(s) lack a valid update reason and source: "
                    f"{missing_reasons}"
                )

            final_decisions = [
                decisions_by_field[field]
                for field in final_changes
            ]
            return guarded_result, final_decisions

        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[enrich_representative] {representative['iro_uid']} "
                f"attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}); retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Enrichment failed for {representative['iro_uid']} after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Flat CSV output matching the secondary-extraction review sheet
# ---------------------------------------------------------------------------

def channel_map(
    financial_effects: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["channel"]): item
        for item in financial_effects
        if item.get("channel")
    }


def flatten_final_results(
    register: pd.DataFrame,
    structured_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create one CSV row per deduplicated IRO, aligned with the baseline
    secondary-extraction CSV. Unique and related rows retain their original
    context; representative rows contain the enriched DMA fields."""
    by_id = {str(record["iro_id"]): record for record in structured_records}
    output_rows: list[dict[str, Any]] = []

    context_columns = [
        "iro_uid",
        "filename",
        "doc_id",
        "document_type",
        "standard",
        "esrs_topic",
        "esrs_subtopic",
        "esrs_sub_subtopic",
        "iro_type",
        "rule_id",
        "evidence_status",
        "supporting_excerpt",
        "reasoning",
        "merge_group_id",
        "dedup_role",
        "related_iro_ids",
        "dedup_rationale",
    ]

    for _, source_row in register.iterrows():
        iro_id = safe_text(source_row.get("iro_uid"))
        result = by_id.get(iro_id)
        if result is None:
            continue

        row = {
            column: source_row.get(column)
            for column in context_columns
            if column in register.columns
        }
        row.update(
            {
                "dma_iro_status": result.get("iro_status"),
                "iro_status_evidence": result.get("iro_status_evidence"),
                "time_horizons": ";".join(
                    result.get("time_horizons", [])
                ),
                "time_horizon_support": result.get(
                    "time_horizon_support"
                ),
                "time_horizon_evidence": result.get(
                    "time_horizon_evidence"
                ),
                "scale_support": (result.get("scale") or {}).get(
                    "support"
                ),
                "scale_evidence": (result.get("scale") or {}).get(
                    "evidence"
                ),
                "scale_note": (result.get("scale") or {}).get("note"),
                "scope_support": (result.get("scope") or {}).get(
                    "support"
                ),
                "scope_evidence": (result.get("scope") or {}).get(
                    "evidence"
                ),
                "scope_note": (result.get("scope") or {}).get("note"),
                "irremediability_support": (
                    result.get("irremediability") or {}
                ).get("support"),
                "irremediability_evidence": (
                    result.get("irremediability") or {}
                ).get("evidence"),
                "irremediability_note": (
                    result.get("irremediability") or {}
                ).get("note"),
                "financial_effects_status": result.get(
                    "financial_effects_status"
                ),
                "likelihood_support": (
                    result.get("likelihood") or {}
                ).get("support"),
                "likelihood_evidence": (
                    result.get("likelihood") or {}
                ).get("evidence"),
                "likelihood_note": (
                    result.get("likelihood") or {}
                ).get("note"),
                "field_update_reasons": json.dumps(
                    result.get("field_update_reasons", []),
                    ensure_ascii=False,
                ),
                "updated_fields": ";".join(
                    str(item.get("field_name", ""))
                    for item in result.get("field_update_reasons", [])
                    if item.get("field_name")
                ),
                "field_update_reason_text": " | ".join(
                    f"{item.get('field_name', '')}: {item.get('reason', '')}"
                    for item in result.get("field_update_reasons", [])
                    if item.get("field_name")
                ),
            }
        )

        effects = channel_map(result.get("financial_effects", []))
        for channel in FinancialEffectChannel:
            item = effects.get(channel.value, {})
            row[f"{channel.value}__support"] = item.get("support")
            row[f"{channel.value}__magnitude_evidence"] = item.get(
                "magnitude_evidence"
            )
            row[f"{channel.value}__note"] = item.get("note")

        output_rows.append(row)

    return pd.DataFrame(output_rows)


def save_flat_csv(
    register: pd.DataFrame,
    structured_records: list[dict[str, Any]],
    csv_path: str,
) -> pd.DataFrame:
    flat = flatten_final_results(register, structured_records)
    flat.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(flat)} final DMA result(s) to {csv_path}")
    return flat


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

AUDITED_FIELDS = [
    "iro_status",
    "iro_status_evidence",
    "time_horizons",
    "time_horizon_support",
    "time_horizon_evidence",
    "scale",
    "scope",
    "irremediability",
    "financial_effects_status",
    "financial_effects",
    "likelihood",
]


def changed_fields(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[str]:
    return [
        field
        for field in AUDITED_FIELDS
        if old.get(field) != new.get(field)
    ]


def load_existing_audit(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return load_csv(path)


def upsert_audit_row(path: str, row: dict[str, Any]) -> None:
    existing = load_existing_audit(path)
    if not existing.empty and "representative_iro_id" in existing.columns:
        existing = existing[
            existing["representative_iro_id"].astype(str)
            != str(row["representative_iro_id"])
        ]
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_enrichment(
    baseline_jsonl: str = DEFAULT_BASELINE_JSONL,
    dedup_register: str = DEFAULT_DEDUP_REGISTER,
    merge_mapping: str = DEFAULT_MERGE_MAPPING,
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    jsonl_output: str = DEFAULT_JSONL_OUTPUT,
    csv_output: str = DEFAULT_CSV_OUTPUT,
    audit_output: str = DEFAULT_AUDIT_OUTPUT,
    failure_output: str = DEFAULT_FAILURE_OUTPUT,
    model: str = DEFAULT_MODEL,
    max_representatives: int | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to the environment or .env."
        )

    register = load_csv(dedup_register)
    mapping = load_csv(merge_mapping)
    baseline_records = read_jsonl(baseline_jsonl)
    validate_inputs(register, mapping, baseline_records)

    baseline_by_id = {
        str(record["iro_id"]): record for record in baseline_records
    }

    representatives = register[
        register["dedup_role"].astype(str) == "representative"
    ].copy()
    if max_representatives is not None:
        representatives = representatives.head(max_representatives).copy()

    if overwrite:
        for path in (jsonl_output, csv_output, audit_output, failure_output):
            Path(path).unlink(missing_ok=True)

    representative_ids = set(representatives["iro_uid"].astype(str))

    # Completion is defined by the audit file, not merely by presence in the
    # final JSONL, because the final JSONL also contains unchanged baseline
    # representatives after a failed or partial run.
    existing_audit = load_existing_audit(audit_output)
    if (
        not existing_audit.empty
        and "representative_iro_id" in existing_audit.columns
    ):
        completed_rep_ids = set(
            existing_audit["representative_iro_id"].dropna().astype(str)
        ) & representative_ids
    else:
        completed_rep_ids = set()

    # Recover previously enriched records from an existing complete output.
    # The audit file identifies which representative records are genuine
    # completed enrichments.
    prior_output_by_id: dict[str, dict[str, Any]] = {}
    if Path(jsonl_output).exists():
        prior_output_by_id = {
            str(record["iro_id"]): record
            for record in read_jsonl(jsonl_output)
            if str(record["iro_id"]) in completed_rep_ids
        }

    pending = representatives[
        ~representatives["iro_uid"].astype(str).isin(completed_rep_ids)
    ]

    print(f"Representative IROs selected: {len(representatives)}")
    print(f"Already enriched: {len(completed_rep_ids)}")
    print(f"Pending: {len(pending)}")

    doc_cache = DocumentCache(corpus_dir)
    failures: list[dict[str, Any]] = []

    for position, (_, representative) in enumerate(
        pending.iterrows(), start=1
    ):
        start = time.time()
        representative_id = safe_text(representative.get("iro_uid"))
        group_id = safe_text(representative.get("merge_group_id"))
        baseline = baseline_by_id[representative_id]

        try:
            members = get_group_members(representative, mapping)
            bundles = [
                build_member_bundle(representative, member, doc_cache)
                for _, member in members.iterrows()
            ]
            result, field_updates = enrich_representative(
                representative=representative,
                baseline=baseline,
                bundles=bundles,
                model=model,
            )

            new_record = result.model_dump(mode="json")
            new_record["field_update_reasons"] = [
                decision.model_dump(mode="json")
                for decision in field_updates
            ]
            prior_output_by_id[representative_id] = new_record
            changes = changed_fields(baseline, new_record)

            retrieval_modes = {
                bundle["location_mode"] for bundle in bundles
            }
            source_files = sorted(
                {bundle["source_filename"] for bundle in bundles}
            )

            upsert_audit_row(
                audit_output,
                {
                    "representative_iro_id": representative_id,
                    "merge_group_id": group_id,
                    "member_count": len(members),
                    "source_file_count": len(source_files),
                    "source_filenames": ";".join(source_files),
                    "retrieval_modes": ";".join(sorted(retrieval_modes)),
                    "changed_fields": ";".join(changes),
                    "changed_field_count": len(changes),
                    "field_update_reasons": json.dumps(
                        new_record.get("field_update_reasons", []),
                        ensure_ascii=False,
                    ),
                },
            )

            # Keep the complete final JSONL current after every successful
            # representative so an interrupted run remains fully resumable.
            current_final_records = []
            for record in baseline_records:
                current_record = dict(
                    prior_output_by_id.get(str(record["iro_id"]), record)
                )
                current_record.setdefault("field_update_reasons", [])
                current_final_records.append(current_record)
            write_jsonl_atomic(jsonl_output, current_final_records)

            elapsed = time.time() - start
            print(
                f"[{position}/{len(pending)}] DONE {representative_id} "
                f"| members={len(members)} | changed={len(changes)} "
                f"| {elapsed:.1f}s"
            )

        except Exception as exc:
            elapsed = time.time() - start
            print(
                f"[{position}/{len(pending)}] FAILED "
                f"{representative_id} in {elapsed:.1f}s: {exc}"
            )
            failures.append(
                {
                    "representative_iro_id": representative_id,
                    "merge_group_id": group_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            pd.DataFrame(failures).to_csv(
                failure_output,
                index=False,
                encoding="utf-8-sig",
            )

    # Build final output: baseline unique/related remain unchanged; completed
    # representative enrichments replace their baseline records. A failed
    # representative safely retains its baseline result.
    enriched_by_id = prior_output_by_id

    final_records = []
    for record in baseline_records:
        final_record = dict(
            enriched_by_id.get(str(record["iro_id"]), record)
        )
        final_record.setdefault("field_update_reasons", [])
        final_records.append(final_record)
    write_jsonl_atomic(jsonl_output, final_records)
    final_csv_df = save_flat_csv(register, final_records, csv_output)

    failure_df = pd.DataFrame(failures)
    if failure_df.empty:
        Path(failure_output).unlink(missing_ok=True)

    audit_df = load_existing_audit(audit_output)
    print(
        f"Finished: {len(enriched_by_id)} representative result(s) enriched; "
        f"{len(final_records)} total final DMA result(s); "
        f"{len(failure_df)} failure(s)."
    )
    return audit_df, failure_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich representative DMA evidence using all merged-member "
            "source contexts."
        )
    )
    parser.add_argument(
        "--baseline-jsonl",
        default=DEFAULT_BASELINE_JSONL,
    )
    parser.add_argument(
        "--dedup-register",
        default=DEFAULT_DEDUP_REGISTER,
    )
    parser.add_argument(
        "--merge-mapping",
        default=DEFAULT_MERGE_MAPPING,
    )
    parser.add_argument(
        "--company",
        default=COMPANY,
        help=(
            "Company tag used to derive default file names/paths, e.g. "
            "sap, puma. Individual --corpus-dir/--jsonl-output/etc. flags "
            "still override this on a per-path basis."
        ),
    )
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-representatives",
        type=int,
        default=None,
        help="Optional small test run, e.g. --max-representatives 3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior enrichment outputs and start again.",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the structured-output schema and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_schema:
        print(json.dumps(DMAScoringEvidence.model_json_schema(), indent=2))
        return

    run_enrichment(
        baseline_jsonl=args.baseline_jsonl,
        dedup_register=args.dedup_register,
        merge_mapping=args.merge_mapping,
        corpus_dir=args.corpus_dir,
        jsonl_output=args.jsonl_output,
        csv_output=args.csv_output,
        audit_output=args.audit_output,
        failure_output=args.failure_output,
        model=args.model,
        max_representatives=args.max_representatives,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
