"""
Secondary evidence extraction for Double Materiality Assessment (DMA).

Purpose
-------
Takes the deduplicated IRO register produced by the first extraction pipeline
(e.g. sap_iro_dedup_register.csv), re-opens the relevant company document for
each IRO, and extracts only the evidence needed for later DMA scoring.

The script does NOT assign 1-5 scores. It produces auditable evidence for:
- IRO status and time horizon;
- scale and scope for positive impacts;
- scale, scope and irremediability for negative impacts;
- likelihood for potential impacts, risks and opportunities;
- supported financial-effect channels for risks and opportunities.

Structured output is enforced with Pydantic through client.responses.parse().
Outputs are written incrementally so the run can be resumed safely.

Default inputs
--------------
- sap_iro_dedup_register.csv
- the source-document PDF directory used in the first extraction

Default outputs
---------------
- sap_dma_secondary_extraction.jsonl  (full structured audit record)
- sap_dma_secondary_extraction.csv    (flattened review sheet)
- sap_dma_secondary_failures.csv      (rows that failed after retries)

Requires
--------
pip install openai pydantic pandas pdfplumber python-dotenv
and OPENAI_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any

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

DEFAULT_INPUT_CSV = f"{COMPANY}_iro_dedup_register.csv"
DEFAULT_CORPUS_DIR = corpus_dir_for(COMPANY)
DEFAULT_JSONL_OUTPUT = f"{COMPANY}_dma_secondary_extraction.jsonl"
DEFAULT_CSV_OUTPUT = f"{COMPANY}_dma_secondary_extraction.csv"
DEFAULT_FAILURE_OUTPUT = f"{COMPANY}_dma_secondary_failures.csv"

DEFAULT_MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
MAX_DOCUMENT_CHARS = 200_000

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
    magnitude_evidence: str | None = Field(
        default=None,
        description=(
            "A concise, document-grounded quotation or close extract showing the "
            "potential gross financial effect through this channel."
        ),
    )
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


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an ESRS double-materiality evidence extraction expert.

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

Return exactly one DMAScoringEvidence object. The iro_id and iro_type must exactly match the supplied values."""


def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def build_user_prompt(row: pd.Series, document_text: str) -> str:
    context = {
        "iro_id": row["iro_id"],
        "iro_type": row["iro_type"],
        "rule_id": safe_text(row.get("rule_id")),
        "standard": safe_text(row.get("standard")),
        "esrs_topic": safe_text(row.get("esrs_topic")),
        "esrs_subtopic": safe_text(row.get("esrs_subtopic")),
        "esrs_sub_subtopic": safe_text(row.get("esrs_sub_subtopic")),
        "source_filename": safe_text(row.get("filename")),
        "existing_supporting_excerpt": safe_text(
            row.get("supporting_excerpt")
        ),
        "existing_reasoning": safe_text(row.get("reasoning")),
        "dedup_rationale": safe_text(row.get("dedup_rationale")),
        "related_iro_ids": safe_text(row.get("related_iro_ids")),
    }
    return (
        "Existing IRO context:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + '\n\nFull relevant company document text:\n"""\n'
        + document_text
        + '\n"""'
    )


# ---------------------------------------------------------------------------
# Input preparation and document loading
# ---------------------------------------------------------------------------

REQUIRED_INPUT_COLUMNS = {"filename", "iro_type", "rule_id"}
VALID_IRO_TYPES = {item.value for item in IROType}


def load_iro_register(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing)}"
        )

    invalid_types = sorted(
        set(df["iro_type"].dropna().astype(str)) - VALID_IRO_TYPES
    )
    if invalid_types:
        raise ValueError(f"Unsupported iro_type value(s): {invalid_types}")

    # Prefer the stable identifier created by Part 7. Fall back to rule/source.
    if "iro_uid" in df.columns:
        df["iro_id"] = df["iro_uid"].astype(str)
    elif "iro_id" not in df.columns:
        df["iro_id"] = [
            f"{safe_text(row.get('rule_id'))}__{i + 1}"
            for i, (_, row) in enumerate(df.iterrows())
        ]

    if df["iro_id"].duplicated().any():
        duplicates = df.loc[df["iro_id"].duplicated(False), "iro_id"].tolist()
        raise ValueError(
            "iro_id must be unique in the secondary-extraction input. "
            f"Duplicate value(s): {duplicates[:10]}"
        )

    return df.reset_index(drop=True)


class DocumentCache:
    def __init__(self, corpus_dir: str, max_chars: int = MAX_DOCUMENT_CHARS):
        self.corpus_dir = Path(corpus_dir)
        self.max_chars = max_chars
        self._cache: dict[str, str] = {}

    def get(self, filename: str) -> str:
        if filename not in self._cache:
            path = self.corpus_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Source document not found: {path}")
            self._cache[filename] = self._extract(path)
        return self._cache[filename]

    def _extract(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(pages).strip()
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            raise ValueError(
                f"Unsupported source format {suffix!r} for {path.name}; "
                "supported formats are PDF, TXT and MD."
            )

        if not text:
            raise ValueError(
                f"No extractable text in {path.name}; the file may require OCR."
            )

        if len(text) > self.max_chars:
            print(
                f"[DocumentCache] warning: {path.name} has {len(text):,} "
                f"characters; truncating to {self.max_chars:,}."
            )
            text = text[: self.max_chars]

        return text


# ---------------------------------------------------------------------------
# LLM call and deterministic post-validation
# ---------------------------------------------------------------------------

def make_not_disclosed_dimension(note: str) -> DimensionEvidence:
    """Create a standard missing-evidence result for an applicable dimension."""
    return DimensionEvidence(
        support=EvidenceSupport.not_disclosed,
        evidence=None,
        note=note,
    )


def normalize_dimension(
    dimension: DimensionEvidence | None,
) -> DimensionEvidence | None:
    """
    Ensure that a not_disclosed dimension never retains contradictory
    evidence text.
    """
    if dimension is None:
        return None

    if dimension.support == EvidenceSupport.not_disclosed:
        dimension.evidence = None

    return dimension


def reconcile_gaps(result: DMAScoringEvidence) -> DMAScoringEvidence:
    """
    Deterministically align the parsed output with the applicable fields for
    each IRO type and remove internally inconsistent evidence.

    This function does not make substantive DMA judgments or generate new
    evidence. It only normalizes the model output before it is saved.
    """

    # ------------------------------------------------------------------
    # General evidence consistency
    # ------------------------------------------------------------------

    result.scale = normalize_dimension(result.scale)
    result.scope = normalize_dimension(result.scope)
    result.irremediability = normalize_dimension(result.irremediability)
    result.likelihood = normalize_dimension(result.likelihood)

    # likelihood is required by the schema, but normalize_dimension is typed
    # to permit None for reuse with optional dimensions.
    if result.likelihood is None:
        result.likelihood = make_not_disclosed_dimension(
            "The document does not provide usable evidence about likelihood."
        )

    # Keep time-horizon fields internally consistent.
    if not result.time_horizons:
        result.time_horizon_support = EvidenceSupport.not_disclosed
        result.time_horizon_evidence = None
    elif result.time_horizon_support == EvidenceSupport.not_disclosed:
        # A selected horizon cannot simultaneously be unsupported.
        result.time_horizons = []
        result.time_horizon_evidence = None

    # ------------------------------------------------------------------
    # Positive impacts
    # ------------------------------------------------------------------

    if result.iro_type == IROType.positive_impact:
        if result.scale is None:
            result.scale = make_not_disclosed_dimension(
                "The document does not provide usable evidence about the "
                "intensity or significance of the positive impact."
            )

        if result.scope is None:
            result.scope = make_not_disclosed_dimension(
                "The document does not provide usable evidence about how "
                "widespread the positive impact is."
            )

        # These fields are not applicable to positive impacts.
        result.irremediability = None
        result.financial_effects_status = None
        result.financial_effects = []

    # ------------------------------------------------------------------
    # Negative impacts
    # ------------------------------------------------------------------

    elif result.iro_type == IROType.negative_impact:
        if result.scale is None:
            result.scale = make_not_disclosed_dimension(
                "The document does not provide usable evidence about the "
                "intensity or gravity of the negative impact."
            )

        if result.scope is None:
            result.scope = make_not_disclosed_dimension(
                "The document does not provide usable evidence about how "
                "widespread the negative impact is."
            )

        if result.irremediability is None:
            result.irremediability = make_not_disclosed_dimension(
                "The document does not provide usable evidence about whether "
                "the harm can be remedied or the time and resources required."
            )

        # Financial-effect channels are not assessed for impacts.
        result.financial_effects_status = None
        result.financial_effects = []

    # ------------------------------------------------------------------
    # Risks and opportunities
    # ------------------------------------------------------------------

    elif result.iro_type in {IROType.risk, IROType.opportunity}:
        # Impact-materiality severity dimensions are not applicable.
        result.scale = None
        result.scope = None
        result.irremediability = None

        normalized_effects: list[FinancialChannelEvidence] = []
        seen_channels: set[FinancialEffectChannel] = set()

        for effect in result.financial_effects:
            # Unsupported channels should not be included in the channel list.
            if effect.support == EvidenceSupport.not_disclosed:
                continue

            # A supported channel must contain actual magnitude evidence.
            if not effect.magnitude_evidence:
                continue

            effect.magnitude_evidence = effect.magnitude_evidence.strip()
            if not effect.magnitude_evidence:
                continue

            # Retain at most one result for each mutually exclusive channel.
            if effect.channel in seen_channels:
                continue

            normalized_effects.append(effect)
            seen_channels.add(effect.channel)

        result.financial_effects = normalized_effects

        if not result.financial_effects:
            result.financial_effects_status = EvidenceSupport.not_disclosed
        elif any(
            effect.support == EvidenceSupport.explicit
            for effect in result.financial_effects
        ):
            result.financial_effects_status = EvidenceSupport.explicit
        else:
            result.financial_effects_status = EvidenceSupport.implied

    return result


def extract_dma_evidence(
    row: pd.Series,
    document_text: str,
    model: str = DEFAULT_MODEL,
) -> DMAScoringEvidence:
    prompt = build_user_prompt(row, document_text)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text_format=DMAScoringEvidence,
            )
            if response.output_parsed is None:
                raise ValueError(
                    "Model returned no parsed output (possible refusal)."
                )

            result = response.output_parsed
            if result.iro_id != str(row["iro_id"]):
                raise ValueError(
                    f"Model changed iro_id from {row['iro_id']!r} to "
                    f"{result.iro_id!r}."
                )
            if result.iro_type.value != str(row["iro_type"]):
                raise ValueError(
                    f"Model changed iro_type from {row['iro_type']!r} to "
                    f"{result.iro_type.value!r}."
                )

            return reconcile_gaps(result)

        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[extract_dma_evidence] {row['iro_id']} attempt {attempt} "
                f"failed ({type(exc).__name__}: {exc}); retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Secondary extraction failed for {row['iro_id']} after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Resume and output helpers
# ---------------------------------------------------------------------------

def load_completed_ids(jsonl_path: str) -> set[str]:
    path = Path(jsonl_path)
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(str(record["iro_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_no}: {exc}"
                ) from exc
    return completed


def append_jsonl(path: str, result: DMAScoringEvidence) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def channel_map(financial_effects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["channel"]): item for item in financial_effects}


def flatten_results(
    input_df: pd.DataFrame,
    structured_records: list[dict[str, Any]],
) -> pd.DataFrame:
    by_id = {str(r["iro_id"]): r for r in structured_records}
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

    for _, source_row in input_df.iterrows():
        iro_id = str(source_row["iro_id"])
        result = by_id.get(iro_id)
        if result is None:
            continue

        row = {
            col: source_row.get(col)
            for col in context_columns
            if col in input_df.columns or col == "iro_id"
        }
        row.update(
            {
                "dma_iro_status": result["iro_status"],
                "iro_status_evidence": result.get("iro_status_evidence"),
                "time_horizons": ";".join(result.get("time_horizons", [])),
                "time_horizon_support": result.get("time_horizon_support"),
                "time_horizon_evidence": result.get("time_horizon_evidence"),
                "scale_support": (result.get("scale") or {}).get("support"),
                "scale_evidence": (result.get("scale") or {}).get("evidence"),
                "scale_note": (result.get("scale") or {}).get("note"),
                "scope_support": (result.get("scope") or {}).get("support"),
                "scope_evidence": (result.get("scope") or {}).get("evidence"),
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
                "likelihood_support": (result.get("likelihood") or {}).get(
                    "support"
                ),
                "likelihood_evidence": (result.get("likelihood") or {}).get(
                    "evidence"
                ),
                "likelihood_note": (result.get("likelihood") or {}).get(
                    "note"
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
    input_df: pd.DataFrame,
    jsonl_path: str,
    csv_path: str,
) -> None:
    records = read_jsonl(jsonl_path)
    flat = flatten_results(input_df, records)
    flat.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(flat)} completed IRO(s) to {csv_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_secondary_extraction(
    input_csv: str = DEFAULT_INPUT_CSV,
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    jsonl_output: str = DEFAULT_JSONL_OUTPUT,
    csv_output: str = DEFAULT_CSV_OUTPUT,
    failure_output: str = DEFAULT_FAILURE_OUTPUT,
    model: str = DEFAULT_MODEL,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    df = load_iro_register(input_csv)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if overwrite:
        for path in (jsonl_output, csv_output, failure_output):
            Path(path).unlink(missing_ok=True)

    completed_ids = load_completed_ids(jsonl_output)
    pending = df[~df["iro_id"].astype(str).isin(completed_ids)]

    print(f"Input IROs: {len(df)}")
    print(f"Already completed: {len(df) - len(pending)}")
    print(f"Pending: {len(pending)}")

    doc_cache = DocumentCache(corpus_dir)
    failures: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(pending.iterrows(), start=1):
        start = time.time()
        iro_id = str(row["iro_id"])
        filename = safe_text(row["filename"])

        try:
            document_text = doc_cache.get(filename)
            result = extract_dma_evidence(row, document_text, model=model)
            append_jsonl(jsonl_output, result)
            elapsed = time.time() - start
            print(
                f"[{position}/{len(pending)}] DONE {iro_id} "
                f"({row['iro_type']}) in {elapsed:.1f}s"
            )
        except Exception as exc:  # continue the corpus run; preserve audit trail
            elapsed = time.time() - start
            print(
                f"[{position}/{len(pending)}] FAILED {iro_id} in "
                f"{elapsed:.1f}s: {exc}"
            )
            failures.append(
                {
                    "iro_id": iro_id,
                    "filename": filename,
                    "rule_id": safe_text(row.get("rule_id")),
                    "iro_type": safe_text(row.get("iro_type")),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

        # Keep the review CSV current even if the process is interrupted.
        save_flat_csv(df, jsonl_output, csv_output)
        if failures:
            pd.DataFrame(failures).to_csv(
                failure_output, index=False, encoding="utf-8-sig"
            )

    save_flat_csv(df, jsonl_output, csv_output)
    failure_df = pd.DataFrame(failures)
    if not failure_df.empty:
        failure_df.to_csv(failure_output, index=False, encoding="utf-8-sig")
        print(f"Saved {len(failure_df)} failure(s) to {failure_output}")
    else:
        Path(failure_output).unlink(missing_ok=True)

    completed_df = pd.read_csv(csv_output, encoding="utf-8-sig")
    print(
        f"Finished: {len(completed_df)} completed, "
        f"{len(failure_df)} failed."
    )
    return completed_df, failure_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured secondary DMA evidence extraction."
    )
    parser.add_argument(
        "--company",
        default=COMPANY,
        help=(
            "Company tag used to derive default file names/paths, e.g. "
            "sap, puma. Individual --input-csv/--corpus-dir/etc. flags "
            "still override this on a per-path basis."
        ),
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional small test run, e.g. --max-rows 3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs and start from the beginning.",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the structured-output JSON schema and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_schema:
        print(json.dumps(DMAScoringEvidence.model_json_schema(), indent=2))
        return

    run_secondary_extraction(
        input_csv=args.input_csv,
        corpus_dir=args.corpus_dir,
        jsonl_output=args.jsonl_output,
        csv_output=args.csv_output,
        failure_output=args.failure_output,
        model=args.model,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
