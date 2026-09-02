"""
Filter material ESRS subtopics and material IROs.

Materiality rule
----------------
1. An individual IRO is material when final_score >= THRESHOLD.
2. A subtopic is material when it contains at least one material IRO.
   This is equivalent to:
       maximum IRO score within the subtopic >= THRESHOLD

Inputs
------
- sap_dma_dimension_scores.jsonl
- sap_dma_secondary_extraction_enriched.csv

Outputs
-------
- material_subtopics.csv
- material_iros.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from company_config import get_company

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma

THRESHOLD = 15.0

STANDARD_ORDER = {
    "E1": 1,
    "E2": 2,
    "E3": 3,
    "E4": 4,
    "E5": 5,
    "S1": 6,
    "S2": 7,
    "S3": 8,
    "S4": 9,
    "G1": 10,
}


def load_scores(path: Path) -> pd.DataFrame:
    """Load IRO final scores from the scoring JSONL file."""
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if "iro_id" not in record or "final_score" not in record:
                raise ValueError(
                    f"Missing iro_id or final_score on line {line_number}."
                )

            records.append(
                {
                    "iro_id": str(record["iro_id"]),
                    "score_iro_type": record.get("iro_type"),
                    "final_score": float(record["final_score"]),
                }
            )

    scores = pd.DataFrame(records)

    if scores.empty:
        raise ValueError("The scoring JSONL file contains no records.")

    if scores["iro_id"].duplicated().any():
        duplicates = scores.loc[
            scores["iro_id"].duplicated(keep=False), "iro_id"
        ].tolist()
        raise ValueError(f"Duplicate iro_id values in scoring file: {duplicates[:10]}")

    return scores


def derive_impact_direction(iro_type: str) -> str:
    """
    Derive negative/positive direction.

    Risks and opportunities are financial IROs, so impact direction is
    recorded as not_applicable rather than forcing them into negative/positive.
    """
    if iro_type == "negative_impact":
        return "negative"
    if iro_type == "positive_impact":
        return "positive"
    return "not_applicable"


def prepare_materiality_results(
    scores_path: Path,
    enriched_path: Path,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge source files and create material subtopic and IRO outputs."""
    scores = load_scores(scores_path)
    enriched = pd.read_csv(enriched_path, encoding="utf-8-sig")

    required_columns = {
        "iro_uid",
        "standard",
        "esrs_topic",
        "esrs_subtopic",
        "esrs_sub_subtopic",
        "iro_type",
        "dma_iro_status",
        "time_horizons",
        "supporting_excerpt",
        "reasoning",
    }
    missing_columns = required_columns - set(enriched.columns)

    if missing_columns:
        raise ValueError(
            "The enriched CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if enriched["iro_uid"].duplicated().any():
        raise ValueError("The enriched CSV contains duplicate iro_uid values.")

    merged = enriched.merge(
        scores,
        left_on="iro_uid",
        right_on="iro_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(scores) or len(merged) != len(enriched):
        score_ids = set(scores["iro_id"])
        enriched_ids = set(enriched["iro_uid"].astype(str))

        only_in_scores = sorted(score_ids - enriched_ids)
        only_in_enriched = sorted(enriched_ids - score_ids)

        raise ValueError(
            "The two files do not match one-to-one. "
            f"Only in scoring file: {only_in_scores[:10]}; "
            f"only in enriched file: {only_in_enriched[:10]}."
        )

    # Use Standard + Topic + Subtopic as the grouping key so that identically
    # named subtopics under different ESRS standards are not incorrectly merged.
    group_columns = ["standard", "esrs_topic", "esrs_subtopic"]

    subtopic_summary = (
        merged.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            subtopic_max_score=("final_score", "max"),
            subtopic_average_score=("final_score", "mean"),
            total_iro_count=("iro_uid", "count"),
            material_iro_count=(
                "final_score",
                lambda values: int((values >= threshold).sum()),
            ),
        )
    )

    subtopic_summary["material_subtopic"] = (
        subtopic_summary["subtopic_max_score"] >= threshold
    )

    material_subtopics = subtopic_summary[
        subtopic_summary["material_subtopic"]
    ].copy()

    # Only IROs meeting the same threshold are classified as material IROs.
    material_iros = merged[merged["final_score"] >= threshold].copy()

    material_iros["impact_direction"] = material_iros["iro_type"].map(
        derive_impact_direction
    )
    material_iros["actual_or_potential"] = material_iros[
        "dma_iro_status"
    ].fillna("unclear")
    material_iros["time_horizon"] = material_iros[
        "time_horizons"
    ].fillna("not_disclosed")

    # Add subtopic-level maximum score and counts to each material IRO row.
    material_iros = material_iros.merge(
        material_subtopics[
            group_columns
            + [
                "subtopic_max_score",
                "subtopic_average_score",
                "total_iro_count",
                "material_iro_count",
            ]
        ],
        on=group_columns,
        how="left",
        validate="many_to_one",
    )

    # Explicit ESRS ordering from E1 to G1.
    material_subtopics["_standard_order"] = (
        material_subtopics["standard"].map(STANDARD_ORDER).fillna(999)
    )
    material_iros["_standard_order"] = (
        material_iros["standard"].map(STANDARD_ORDER).fillna(999)
    )

    material_subtopics = material_subtopics.sort_values(
        [
            "_standard_order",
            "esrs_topic",
            "esrs_subtopic",
        ],
        ascending=[True, True, True],
        kind="stable",
    ).drop(columns="_standard_order")

    material_iros = material_iros.sort_values(
        [
            "_standard_order",
            "esrs_topic",
            "esrs_subtopic",
            "esrs_sub_subtopic",
            "final_score",
            "iro_uid",
        ],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    ).drop(columns="_standard_order")

    material_subtopics["subtopic_max_score"] = (
        material_subtopics["subtopic_max_score"].round(2)
    )
    material_subtopics["subtopic_average_score"] = (
        material_subtopics["subtopic_average_score"].round(2)
    )
    material_iros["final_score"] = material_iros["final_score"].round(2)
    material_iros["subtopic_max_score"] = (
        material_iros["subtopic_max_score"].round(2)
    )
    material_iros["subtopic_average_score"] = (
        material_iros["subtopic_average_score"].round(2)
    )

    subtopic_output_columns = [
        "standard",
        "esrs_topic",
        "esrs_subtopic",
        "subtopic_max_score",
        "subtopic_average_score",
        "total_iro_count",
        "material_iro_count",
        "material_subtopic",
    ]

    iro_output_columns = [
        "standard",
        "esrs_topic",
        "esrs_subtopic",
        "esrs_sub_subtopic",
        "iro_uid",
        "iro_type",
        "impact_direction",
        "actual_or_potential",
        "time_horizon",
        "final_score",
        "subtopic_max_score",
        "subtopic_average_score",
        "total_iro_count",
        "material_iro_count",
        "supporting_excerpt",
        "reasoning",
    ]

    return (
        material_subtopics[subtopic_output_columns].reset_index(drop=True),
        material_iros[iro_output_columns].reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter material subtopics and material IROs using a common "
            "score threshold and the maximum-score subtopic rule."
        )
    )
    parser.add_argument(
        "--company",
        default=COMPANY,
        help=(
            "Company tag used to derive default --scores/--enriched "
            "file names, e.g. sap, puma."
        ),
    )
    parser.add_argument(
        "--scores",
        default=f"{COMPANY}_dma_dimension_scores.jsonl",
        help="Path to the IRO scoring JSONL file.",
    )
    parser.add_argument(
        "--enriched",
        default=f"{COMPANY}_dma_secondary_extraction_enriched.csv",
        help="Path to the enriched IRO CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory in which output CSV files will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Materiality threshold. Default: 15.",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    enriched_path = Path(args.enriched)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    material_subtopics, material_iros = prepare_materiality_results(
        scores_path=scores_path,
        enriched_path=enriched_path,
        threshold=args.threshold,
    )

    subtopic_output = output_dir / "material_subtopics.csv"
    iro_output = output_dir / "material_iros.csv"

    material_subtopics.to_csv(
        subtopic_output,
        index=False,
        encoding="utf-8-sig",
    )
    material_iros.to_csv(
        iro_output,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Threshold: {args.threshold:g}")
    print(f"Material subtopics: {len(material_subtopics)}")
    print(f"Material IROs: {len(material_iros)}")
    print(f"Saved: {subtopic_output}")
    print(f"Saved: {iro_output}")


if __name__ == "__main__":
    main()
