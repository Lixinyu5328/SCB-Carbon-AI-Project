"""
Double Materiality Assessment — Dimension Scoring

Purpose
-------
Scores each required DMA dimension on a 1–5 scale using:
1. company evidence extracted in the enriched secondary-extraction JSONL;
2. the exact scoring criteria in DMA Plan.docx; and
3. cautious assumption-based estimation when a required field is not disclosed.

Important methodological limitation
------------------------------------
An assumption-based score is an analytical estimate only. It must not be
interpreted as a company-disclosed fact, an objective value reported by
the company, or a definitive conclusion about the undertaking.

Expected inputs (defaults shown for --company sap; pass --company <name>
to derive these for another company, e.g. puma)
--------------------------------------------------------------------------
1. sap_dma_secondary_extraction_enriched.jsonl
2. sap_iro_dedup_register.csv

Outputs
-------
1. sap_dma_dimension_scores.jsonl
2. sap_dma_dimension_scores.csv

Each individual model-generated dimension score contains exactly four fields:
    dimension
    score
    score_basis
    reasoning

The outer output record additionally retains iro_id and iro_type so that the
scores can be joined back to the corresponding IRO.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

from company_config import get_company


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma
COMPANY_DISPLAY_NAME = COMPANY.upper()  # used inside LLM prompt text below

DEFAULT_EVIDENCE_PATH = f"{COMPANY}_dma_secondary_extraction_enriched.jsonl"
DEFAULT_REGISTER_PATH = f"{COMPANY}_iro_dedup_register.csv"
DEFAULT_OUTPUT_JSONL = f"{COMPANY}_dma_dimension_scores.jsonl"
DEFAULT_OUTPUT_CSV = f"{COMPANY}_dma_dimension_scores.csv"

MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

RETRYABLE_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    ValueError,
)

FINANCIAL_CHANNEL_LABELS = {
    "revenue_and_business_growth": "Revenue and Business Growth",
    "costs_and_profitability": "Costs and Profitability",
    "assets_liabilities_and_financial_position": (
        "Assets, Liabilities and Financial Position"
    ),
    "cash_flow_and_liquidity": "Cash Flow and Liquidity",
    "access_to_finance_and_cost_of_capital": (
        "Access to Finance and Cost of Capital"
    ),
    "business_continuity_and_strategic_viability": (
        "Business Continuity and Strategic Viability"
    ),
}


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class ScoreBasis(str, Enum):
    explicit = "explicit"
    implied = "implied"
    assumption_based = "assumption_based"
    methodology_rule = "methodology_rule"


class DimensionScore(BaseModel):
    """The four fields retained for every individual dimension score."""

    dimension: str
    score: int = Field(ge=1, le=5)
    score_basis: ScoreBasis
    reasoning: str


class ScoredIRO(BaseModel):
    """
    Wrapper used only to preserve the link between the four-field scores
    and the original IRO and to store the final calculated DMA score.
    """

    iro_id: str
    iro_type: str
    dimension_scores: list[DimensionScore]
    final_score: float


class InferredFinancialChannels(BaseModel):
    """One or more assumption-based financial channels selected from the DMA Plan."""

    channels: list[str]


# ---------------------------------------------------------------------------
# Exact scoring criteria from DMA Planpy.docx
# Do not paraphrase or modify these definitions.
# ---------------------------------------------------------------------------

SCORE_CRITERIA: dict[str, dict[int, str]] = {
    "negative_scale": {
        1: "Negligible impact, causing little or no substantive change",
        2: "Limited, short-term or readily manageable adverse impact",
        3: (
            "Noticeable adverse impact requiring management, mitigation or "
            "remediation"
        ),
        4: (
            "Severe adverse impact causing a major change to health, rights, "
            "environmental conditions or well-being"
        ),
        5: (
            "Extreme or catastrophic adverse impact, including threats to life, "
            "serious violations of fundamental rights or severe destruction of "
            "ecosystems"
        ),
    },
    "positive_scale": {
        1: "Symbolic, marginal or negligible improvement",
        2: "Limited but identifiable improvement",
        3: "Clear and meaningful improvement",
        4: (
            "Major improvement to people, rights, well-being or environmental "
            "conditions"
        ),
        5: (
            "Transformational or fundamental improvement producing exceptionally "
            "significant benefits"
        ),
    },
    "scope": {
        1: "A single individual, isolated event or very limited local area",
        2: "A small group, single location or limited business unit",
        3: "Multiple groups, locations, business units or value-chain nodes",
        4: (
            "A large population, multiple countries, extensive operations or "
            "broad ecological areas"
        ),
        5: (
            "Very widespread or systemic impact affecting an exceptionally large "
            "population, geographic area, ecosystem or value chain"
        ),
    },
    "irremediability": {
        1: "The impact can be rapidly and fully remedied",
        2: (
            "The impact can largely be remedied with limited time and resources"
        ),
        3: (
            "The impact can only be partially remedied, or remediation requires "
            "substantial time or resources"
        ),
        4: (
            "The impact is very difficult to remedy and may have long-term "
            "consequences"
        ),
        5: "The impact is irreversible or cannot realistically be remedied",
    },
    "likelihood": {
        1: (
            "Very unlikely and only conceivable under exceptional circumstances, "
            "with little or no supporting evidence"
        ),
        2: (
            "Unlikely, a credible pathway exists but evidence that it will occur "
            "is limited"
        ),
        3: (
            "Possible, supported by relevant exposure, historical events, trends "
            "or a plausible scenario, but subject to significant uncertainty"
        ),
        4: (
            "Likely, supported by strong evidence, clear trends, repeated exposure "
            "or approved plans"
        ),
        5: (
            "Almost certain or expected to occur, supported by compelling evidence "
            "or an established recurring pattern"
        ),
    },
    "risk_financial_magnitude": {
        1: (
            "No substantial financial effect; immaterial and readily absorbed "
            "without management attention"
        ),
        2: (
            "Limited effect that can be absorbed through routine management and "
            "does not materially affect core objectives"
        ),
        3: (
            "Noticeable effect on a financial channel, requiring active management "
            "and potentially affecting budgets, targets or business-unit performance"
        ),
        4: (
            "Major effect on group-level financial performance, key operations, "
            "strategic objectives, financing capacity or important assets"
        ),
        5: (
            "Effect could threaten business continuity, financial viability, "
            "solvency, major assets, key markets, operating licenses or the "
            "long-term viability of the business model"
        ),
    },
    "opportunity_financial_magnitude": {
        1: (
            "No substantial financial benefit; immaterial and requiring no "
            "management attention"
        ),
        2: (
            "Limited benefit that can be realized through routine management and "
            "does not materially affect core objectives"
        ),
        3: (
            "Noticeable benefit through a financial channel, requiring active "
            "management and potentially improving budgets, targets or business-unit "
            "performance"
        ),
        4: (
            "Major benefit to group-level financial performance, key operations, "
            "strategic objectives, financing capacity or important assets"
        ),
        5: (
            "Benefit could fundamentally strengthen business continuity, financial "
            "viability, major assets, key markets, operating capabilities or the "
            "long-term viability of the business model."
        ),
    },
}


FINANCIAL_CHANNEL_DEFINITIONS = {
    "revenue_and_business_growth": (
        "The effect on sales, customer demand, market share, market access and "
        "future growth."
    ),
    "costs_and_profitability": (
        "The effect on operating costs, capital expenditure, margins and overall "
        "profitability."
    ),
    "assets_liabilities_and_financial_position": (
        "The effect on asset values, impairments, provisions, liabilities and the "
        "company’s financial position."
    ),
    "cash_flow_and_liquidity": (
        "The effect on the amount, timing and stability of cash flows and the "
        "ability to meet short-term obligations."
    ),
    "access_to_finance_and_cost_of_capital": (
        "The effect on the availability, terms and cost of debt, equity, insurance "
        "or other financing."
    ),
    "business_continuity_and_strategic_viability": (
        "The effect on the company’s ability to maintain core operations and "
        "sustain a viable business model."
    ),
}


# ---------------------------------------------------------------------------
# Input loading and joining
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return records


def build_register_lookup(path: str | Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"iro_uid", "supporting_excerpt", "reasoning"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dedup register is missing required columns: {sorted(missing)}"
        )

    df = df.where(pd.notna(df), None)
    return {
        str(row["iro_uid"]): row
        for row in df.to_dict(orient="records")
    }


def build_iro_context(
    evidence_record: dict[str, Any],
    register_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    iro_id = str(evidence_record["iro_id"])
    source = register_lookup.get(iro_id, {})

    description_parts = [
        source.get("esrs_topic"),
        source.get("esrs_subtopic"),
        source.get("esrs_sub_subtopic"),
    ]
    iro_description = " / ".join(
        str(value) for value in description_parts if value
    )

    return {
        "iro_id": iro_id,
        "iro_type": evidence_record.get("iro_type", ""),
        "iro_status": evidence_record.get("iro_status", "unclear"),
        "iro_description": iro_description or "Not available",
        "supporting_excerpt": source.get("supporting_excerpt") or "",
        "original_reasoning": source.get("reasoning") or "",
    }


# ---------------------------------------------------------------------------
# Criteria and prompt helpers
# ---------------------------------------------------------------------------

def get_score_criteria(
    iro_type: str,
    dimension: str,
) -> dict[int, str]:
    if dimension == "scale":
        if iro_type == "negative_impact":
            return SCORE_CRITERIA["negative_scale"]
        if iro_type == "positive_impact":
            return SCORE_CRITERIA["positive_scale"]
        raise ValueError(f"Scale is not scored for iro_type={iro_type}")

    if dimension == "scope":
        return SCORE_CRITERIA["scope"]

    if dimension == "irremediability":
        return SCORE_CRITERIA["irremediability"]

    if dimension == "likelihood":
        return SCORE_CRITERIA["likelihood"]

    if dimension.startswith("financial_magnitude"):
        if iro_type == "risk":
            return SCORE_CRITERIA["risk_financial_magnitude"]
        if iro_type == "opportunity":
            return SCORE_CRITERIA["opportunity_financial_magnitude"]
        raise ValueError(
            f"Financial magnitude is not scored for iro_type={iro_type}"
        )

    raise ValueError(
        f"No scoring criteria for iro_type={iro_type}, dimension={dimension}"
    )


def format_score_criteria(criteria: dict[int, str]) -> str:
    return "\n".join(
        f"{score}: {definition}"
        for score, definition in criteria.items()
    )


def normalize_support(value: Any) -> str:
    support = str(value or "").strip().lower()
    if support in {"explicit", "implied"}:
        return support
    return "not_disclosed"


def derive_score_basis(support: str) -> ScoreBasis:
    normalized = normalize_support(support)
    if normalized == "explicit":
        return ScoreBasis.explicit
    if normalized == "implied":
        return ScoreBasis.implied
    return ScoreBasis.assumption_based


SCORING_SYSTEM_PROMPT = f"""You are conducting a Double Materiality Assessment for {COMPANY_DISPLAY_NAME}.

The existence of the specific impact, risk or opportunity supplied by the user has already been verified. Assign one integer score from 1 to 5 for the requested dimension, using the supplied scoring criteria exactly as written.

For explicit or implied company evidence:
- Base the score on the supplied evidence.
- Do not add unsupported company-specific facts.

For a field marked not_disclosed:
- Use the verified existence of the IRO, the available company evidence, and cautious knowledge of {COMPANY_DISPLAY_NAME}'s industry, business model and value chain.
- Do not treat absence of disclosure as evidence of a low score.
- Do not invent {COMPANY_DISPLAY_NAME}-specific incidents, quantities, affected populations, geographic coverage, financial figures or outcomes.
- Clearly identify the score as assumption_based.
- Use conservative assumptions where evidence is limited.
- The score is an analytical estimate only, not a company-disclosed fact, an objective value reported by {COMPANY_DISPLAY_NAME}, or a definitive conclusion.

For scale, do not use the number of employees, countries, suppliers, sites, product lines or value-chain nodes as evidence. These describe scope, not scale.
For risks, score the magnitude of potential financial losses or adverse effects.
For opportunities, score the magnitude of potential financial gains or beneficial effects. Do not increase an opportunity score because implementing the opportunity may create additional costs; such costs are not opportunity benefits.
Select only channels with a direct financial pathway. Do not select a channel merely because an indirect or hypothetical connection can be imagined.

The reasoning must be concise, normally one to three sentences.
Return only the structured four-field result requested by the schema."""


def build_scoring_prompt(
    *,
    context: dict[str, Any],
    dimension: str,
    support: str,
    evidence: str | None,
    note: str | None,
    channel: str | None = None,
) -> str:
    iro_type = str(context["iro_type"])
    criteria = get_score_criteria(iro_type, dimension)
    basis = derive_score_basis(support)

    channel_section = ""
    if channel:
        channel_label = FINANCIAL_CHANNEL_LABELS.get(channel, channel)
        channel_definition = FINANCIAL_CHANNEL_DEFINITIONS.get(channel, "")
        channel_section = (
            f"\nFinancial channel: {channel_label}\n"
            f"Channel definition: {channel_definition}\n"
        )

    return f"""IRO ID: {context['iro_id']}
IRO type: {iro_type}
IRO status: {context['iro_status']}
Verified IRO description: {context['iro_description']}

Original company supporting excerpt:
{context['supporting_excerpt'] or 'Not available'}

Original IRO reasoning:
{context['original_reasoning'] or 'Not available'}

Dimension to score: {dimension}
Evidence support: {normalize_support(support)}
Required score basis: {basis.value}
{channel_section}
Dimension-specific company evidence:
{evidence or 'No company-specific evidence was disclosed for this dimension.'}

Evidence note:
{note or 'No additional note was provided.'}

Scoring criteria:
{format_score_criteria(criteria)}

Select the single most reasonable integer score from 1 to 5.
The output dimension must be exactly: {dimension}
The output score_basis must be exactly: {basis.value}
Return dimension, score, score_basis and concise reasoning only.""".strip()


# ---------------------------------------------------------------------------
# Financial-channel inference when company disclosure is absent
# ---------------------------------------------------------------------------

FINANCIAL_CHANNEL_INFERENCE_SYSTEM_PROMPT = f"""You are conducting a Double Materiality Assessment for {COMPANY_DISPLAY_NAME}.

The existence of the supplied risk or opportunity has already been verified, but the available company documents do not disclose a financial effect channel.

Select one or more channels from the following fixed DMA Plan list that provide the most direct and reasonable financial pathway for this specific IRO:
- revenue_and_business_growth
- costs_and_profitability
- assets_liabilities_and_financial_position
- cash_flow_and_liquidity
- access_to_finance_and_cost_of_capital
- business_continuity_and_strategic_viability

Rules:
- Select only channels with a direct and plausible causal connection to the verified IRO.
- Do not select every channel merely because indirect effects are conceivable.
- Do not invent {COMPANY_DISPLAY_NAME}-specific incidents, amounts, losses, gains or outcomes.
- This is an assumption-based analytical inference, not a company-disclosed fact.
- Return at least one channel and use only the exact channel values listed above.
"""


def infer_financial_channels(
    client: OpenAI,
    context: dict[str, Any],
) -> list[str]:
    allowed_channels = set(FINANCIAL_CHANNEL_DEFINITIONS)
    channel_list = "\n".join(
        f"- {channel}: {definition}"
        for channel, definition in FINANCIAL_CHANNEL_DEFINITIONS.items()
    )
    prompt = f"""IRO ID: {context['iro_id']}
IRO type: {context['iro_type']}
IRO status: {context['iro_status']}
Verified IRO description: {context['iro_description']}

Original company supporting excerpt:
{context['supporting_excerpt'] or 'Not available'}

Original IRO reasoning:
{context['original_reasoning'] or 'Not available'}

Available financial channels:
{channel_list}

Select the one or more channels with the most direct and reasonable financial pathway for this verified IRO."""

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {
                        "role": "system",
                        "content": FINANCIAL_CHANNEL_INFERENCE_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=InferredFinancialChannels,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("Model returned no parsed financial channels.")

            channels: list[str] = []
            for channel in parsed.channels:
                normalized = str(channel).strip()
                if normalized in allowed_channels and normalized not in channels:
                    channels.append(normalized)

            if not channels:
                raise ValueError(
                    "Model returned no valid financial channel from the DMA Plan list."
                )
            return channels

        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_seconds = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[infer_financial_channels] {context['iro_id']} "
                f"attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}); "
                f"retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Financial-channel inference failed for {context['iro_id']} "
        f"after {MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# OpenAI scoring
# ---------------------------------------------------------------------------

def score_dimension(
    client: OpenAI,
    *,
    context: dict[str, Any],
    dimension: str,
    support: str,
    evidence: str | None,
    note: str | None,
    channel: str | None = None,
) -> DimensionScore:
    prompt = build_scoring_prompt(
        context=context,
        dimension=dimension,
        support=support,
        evidence=evidence,
        note=note,
        channel=channel,
    )
    expected_basis = derive_score_basis(support)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text_format=DimensionScore,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("Model returned no parsed output.")

            if parsed.dimension != dimension:
                raise ValueError(
                    f"Model returned dimension={parsed.dimension!r}; "
                    f"expected {dimension!r}."
                )
            if parsed.score_basis != expected_basis:
                raise ValueError(
                    f"Model returned score_basis={parsed.score_basis.value!r}; "
                    f"expected {expected_basis.value!r}."
                )
            return parsed

        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_seconds = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[score_dimension] {context['iro_id']} / {dimension} "
                f"attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}); "
                f"retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Scoring failed for {context['iro_id']} / {dimension} "
        f"after {MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Required dimensions by IRO type
# ---------------------------------------------------------------------------

def get_evidence_block(
    record: dict[str, Any],
    field_name: str,
) -> tuple[str, str | None, str | None]:
    block = record.get(field_name)

    if not isinstance(block, dict):
        return "not_disclosed", None, None

    return (
        normalize_support(block.get("support")),
        block.get("evidence"),
        block.get("note"),
    )


def fixed_actual_likelihood() -> DimensionScore:
    return DimensionScore(
        dimension="likelihood",
        score=5,
        score_basis=ScoreBasis.methodology_rule,
        reasoning=(
            "The IRO is classified as actual; under the DMA Plan methodology, "
            "the likelihood score for an actual impact is fixed at 5."
        ),
    )


def score_impact_record(
    client: OpenAI,
    record: dict[str, Any],
    context: dict[str, Any],
) -> list[DimensionScore]:
    iro_type = str(record["iro_type"])
    dimensions = ["scale", "scope"]

    if iro_type == "negative_impact":
        dimensions.append("irremediability")

    results: list[DimensionScore] = []

    for dimension in dimensions:
        support, evidence, note = get_evidence_block(record, dimension)
        results.append(
            score_dimension(
                client,
                context=context,
                dimension=dimension,
                support=support,
                evidence=evidence,
                note=note,
            )
        )

    if str(record.get("iro_status", "")).lower() == "actual":
        results.append(fixed_actual_likelihood())
    else:
        support, evidence, note = get_evidence_block(record, "likelihood")
        results.append(
            score_dimension(
                client,
                context=context,
                dimension="likelihood",
                support=support,
                evidence=evidence,
                note=note,
            )
        )

    return results


def score_financial_record(
    client: OpenAI,
    record: dict[str, Any],
    context: dict[str, Any],
) -> list[DimensionScore]:
    results: list[DimensionScore] = []
    effects = record.get("financial_effects") or []

    if effects:
        for effect in effects:
            channel = str(effect.get("channel", "")).strip()
            if not channel:
                continue

            dimension = f"financial_magnitude::{channel}"
            support = normalize_support(
                effect.get("support")
                or record.get("financial_effects_status")
            )
            results.append(
                score_dimension(
                    client,
                    context=context,
                    dimension=dimension,
                    support=support,
                    evidence=effect.get("magnitude_evidence"),
                    note=effect.get("note"),
                    channel=channel,
                )
            )
    else:
        # No financial channel was disclosed. Select the one or more most direct
        # channels from the six DMA Plan channels, then score each selected
        # channel on an assumption-based basis.
        inferred_channels = infer_financial_channels(client, context)
        for channel in inferred_channels:
            dimension = f"financial_magnitude::{channel}"
            results.append(
                score_dimension(
                    client,
                    context=context,
                    dimension=dimension,
                    support="not_disclosed",
                    evidence=None,
                    note=(
                        "The company documents did not disclose a financial effect "
                        "channel or channel-specific magnitude. This channel was "
                        "selected as the most direct reasonable pathway from the "
                        "verified IRO and is scored as an assumption-based estimate."
                    ),
                    channel=channel,
                )
            )

    support, evidence, note = get_evidence_block(record, "likelihood")
    results.append(
        score_dimension(
            client,
            context=context,
            dimension="likelihood",
            support=support,
            evidence=evidence,
            note=note,
        )
    )

    return results


def calculate_final_score(
    iro_type: str,
    dimension_scores: list[DimensionScore],
) -> float:
    score_lookup = {
        item.dimension: item.score
        for item in dimension_scores
    }

    likelihood = score_lookup.get("likelihood")
    if likelihood is None:
        raise ValueError("Likelihood score is required to calculate final_score.")

    if iro_type == "positive_impact":
        scale = score_lookup.get("scale")
        scope = score_lookup.get("scope")
        if scale is None or scope is None:
            raise ValueError(
                "Scale and scope are required for positive-impact final_score."
            )
        benefit = 0.6 * scale + 0.4 * scope
        if scale == 5 or scope == 5:
            benefit = max(benefit, 4.0)
        return round(benefit * likelihood, 2)

    if iro_type == "negative_impact":
        scale = score_lookup.get("scale")
        scope = score_lookup.get("scope")
        irremediability = score_lookup.get("irremediability")
        if scale is None or scope is None or irremediability is None:
            raise ValueError(
                "Scale, scope and irremediability are required for "
                "negative-impact final_score."
            )
        severity = 0.4 * scale + 0.3 * scope + 0.3 * irremediability
        if 5 in {scale, scope, irremediability}:
            severity = max(severity, 4.0)
        return round(severity * likelihood, 2)

    if iro_type in {"risk", "opportunity"}:
        channel_scores = [
            item.score
            for item in dimension_scores
            if item.dimension.startswith("financial_magnitude")
        ]
        if not channel_scores:
            raise ValueError(
                "At least one financial-magnitude channel score is required."
            )
        magnitude = max(channel_scores)
        return round(magnitude * likelihood, 2)

    raise ValueError(f"Unsupported iro_type={iro_type!r}")


def score_one_iro(
    client: OpenAI,
    record: dict[str, Any],
    register_lookup: dict[str, dict[str, Any]],
) -> ScoredIRO:
    context = build_iro_context(record, register_lookup)
    iro_type = str(record.get("iro_type", ""))

    if iro_type in {"negative_impact", "positive_impact"}:
        dimension_scores = score_impact_record(client, record, context)
    elif iro_type in {"risk", "opportunity"}:
        dimension_scores = score_financial_record(client, record, context)
    else:
        raise ValueError(
            f"Unsupported iro_type={iro_type!r} for iro_id={context['iro_id']}"
        )

    final_score = calculate_final_score(iro_type, dimension_scores)

    return ScoredIRO(
        iro_id=context["iro_id"],
        iro_type=iro_type,
        dimension_scores=dimension_scores,
        final_score=final_score,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def flatten_scored_iros(scored: list[ScoredIRO]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in scored:
        for dimension_score in item.dimension_scores:
            rows.append(
                {
                    "iro_id": item.iro_id,
                    "iro_type": item.iro_type,
                    **dimension_score.model_dump(mode="json"),
                    "final_score": item.final_score,
                }
            )

    return pd.DataFrame(rows)


def append_jsonl(path: str | Path, item: ScoredIRO) -> None:
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n"
        )


def load_completed_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()

    completed: set[str] = set()
    with output.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["iro_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def jsonl_to_csv(
    jsonl_path: str | Path,
    csv_path: str | Path,
) -> None:
    scored_records = [
        ScoredIRO.model_validate(record)
        for record in load_jsonl(jsonl_path)
    ]
    flatten_scored_iros(scored_records).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def remove_existing_records_by_type(
    path: str | Path,
    iro_types: set[str],
) -> int:
    """Remove existing output records for selected IRO types and preserve all others."""
    output = Path(path)
    if not output.exists():
        return 0

    kept: list[dict[str, Any]] = []
    removed = 0
    for record in load_jsonl(output):
        if str(record.get("iro_type", "")) in iro_types:
            removed += 1
        else:
            kept.append(record)

    with output.open("w", encoding="utf-8") as file:
        for record in kept:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return removed


def run_scoring(
    *,
    evidence_path: str,
    register_path: str,
    output_jsonl: str,
    output_csv: str,
    limit: int | None = None,
    overwrite: bool = False,
    iro_type_filter: str | None = None,
    replace_selected: bool = False,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your environment or .env file."
        )

    client = OpenAI(api_key=api_key)
    evidence_records = load_jsonl(evidence_path)
    register_lookup = build_register_lookup(register_path)

    output_jsonl_path = Path(output_jsonl)
    if overwrite and output_jsonl_path.exists():
        output_jsonl_path.unlink()

    if iro_type_filter is not None:
        evidence_records = [
            record
            for record in evidence_records
            if str(record.get("iro_type", "")) == iro_type_filter
        ]

    if replace_selected:
        if iro_type_filter is None:
            raise ValueError(
                "--replace-selected requires --iro-type so that unselected "
                "IRO categories remain unchanged."
            )
        removed = remove_existing_records_by_type(
            output_jsonl_path,
            {iro_type_filter},
        )
        print(
            f"Removed {removed} existing {iro_type_filter} record(s) "
            "from the JSONL output before rescoring."
        )

    completed_ids = load_completed_ids(output_jsonl_path)
    pending = [
        record
        for record in evidence_records
        if str(record.get("iro_id")) not in completed_ids
    ]

    if limit is not None:
        pending = pending[:limit]

    print(
        f"Loaded {len(evidence_records)} IROs; "
        f"{len(completed_ids)} already completed; "
        f"{len(pending)} pending."
    )

    failed: list[tuple[str, str]] = []

    for index, record in enumerate(pending, start=1):
        iro_id = str(record.get("iro_id", "unknown"))
        try:
            scored = score_one_iro(client, record, register_lookup)
            append_jsonl(output_jsonl_path, scored)
            print(f"[{index}/{len(pending)}] DONE {iro_id}")
        except Exception as exc:
            failed.append((iro_id, str(exc)))
            print(
                f"[{index}/{len(pending)}] FAILED {iro_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    if output_jsonl_path.exists():
        jsonl_to_csv(output_jsonl_path, output_csv)
        print(f"Saved JSONL to {output_jsonl}")
        print(f"Saved flattened CSV to {output_csv}")

    if failed:
        print(f"\n{len(failed)} IRO(s) failed:")
        for iro_id, error in failed:
            print(f"  - {iro_id}: {error}")
    else:
        print("\nAll requested IROs were scored successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Score {COMPANY_DISPLAY_NAME} DMA dimensions using the DMA Plan criteria."
    )
    parser.add_argument(
        "--company",
        default=COMPANY,
        help=(
            "Company tag used to derive default file names and the "
            "company name referenced in LLM prompts, e.g. sap, puma. "
            "Individual --evidence/--register/etc. flags still override "
            "this on a per-path basis."
        ),
    )
    parser.add_argument(
        "--evidence",
        default=DEFAULT_EVIDENCE_PATH,
        help="Path to the enriched secondary-extraction JSONL.",
    )
    parser.add_argument(
        "--register",
        default=DEFAULT_REGISTER_PATH,
        help="Path to the deduplicated IRO register CSV.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Path for structured JSONL output.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path for flattened CSV output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of pending IROs to score for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing JSONL output and start again.",
    )
    parser.add_argument(
        "--iro-type",
        choices=[
            "negative_impact",
            "positive_impact",
            "risk",
            "opportunity",
        ],
        default=None,
        help="Optionally score only one IRO category.",
    )
    parser.add_argument(
        "--replace-selected",
        action="store_true",
        help=(
            "Remove existing output records for --iro-type, preserve all other "
            "categories, and rescore only the selected category."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scoring(
        evidence_path=args.evidence,
        register_path=args.register,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        limit=args.limit,
        overwrite=args.overwrite,
        iro_type_filter=args.iro_type,
        replace_selected=args.replace_selected,
    )


if __name__ == "__main__":
    main()
