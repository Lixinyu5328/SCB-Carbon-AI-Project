import re

import pandas as pd

from company_config import get_company
from part1_taxonomy_mapping import build_rule_index, load_rulebook, load_taxonomy, merge_rule_entries

RULEBOOK_PATH = "IRO_rulebook_Final_1.json"
TAXONOMY_PATH = "Document_taxonomy_1.docx"
COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma

RULEBOOK_JOIN_FIELDS = [
    "standard",
    "esrs_topic",
    "esrs_subtopic",
    "esrs_sub_subtopic",
    "iro_type",
    "iro_status",
    "value_chain_position",
    "time_horizon",
]

OUTPUT_COLUMNS = [
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
]

HIT_STATUSES = ("explicit", "implied")


def build_rulebook_lookup(rulebook_path: str = RULEBOOK_PATH) -> dict[str, dict]:
    standard_rules = load_rulebook(rulebook_path)
    lookup = {}
    for rules in standard_rules.values():
        for rule in rules:
            lookup[rule["rule_id"]] = rule
    return lookup


def enrich_results(results: list[dict], rulebook_lookup: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)

    missing_ids = set(df["rule_id"]) - set(rulebook_lookup.keys())
    if missing_ids:
        print(f"[enrich_results] warning: {len(missing_ids)} rule_id(s) not found in rulebook: {sorted(missing_ids)}")

    rulebook_df = pd.DataFrame(rulebook_lookup.values())[["rule_id"] + RULEBOOK_JOIN_FIELDS]
    merged = df.merge(rulebook_df, on="rule_id", how="left")

    return merged[[c for c in OUTPUT_COLUMNS if c in merged.columns]]


def validate_coverage(df: pd.DataFrame, rule_index: dict[int, dict]) -> None:
    for filename, group in df.groupby("filename"):
        doc_id = group["doc_id"].iloc[0]
        if isinstance(doc_id, str) and "&" in doc_id:
            doc_ids = [int(n) for n in re.findall(r"\d+", doc_id)]
            if all(d in rule_index for d in doc_ids):
                expected = merge_rule_entries(doc_ids, rule_index).get("rule_count")
            else:
                expected = None
        else:
            expected = rule_index.get(doc_id, {}).get("rule_count")
        actual = len(group)
        if expected is not None and actual != expected:
            print(f"[validate_coverage] {filename}: expected {expected} rows, got {actual}")


def save_csv_outputs(df: pd.DataFrame, output_dir: str = ".", prefix: str = f"{COMPANY}_iro_extraction") -> tuple[str, str]:
    full_path = f"{output_dir}/{prefix}_full.csv"
    hits_path = f"{output_dir}/{prefix}_hits.csv"

    df.to_csv(full_path, index=False, encoding="utf-8-sig")

    hits_df = df[df["evidence_status"].isin(HIT_STATUSES)]
    hits_df.to_csv(hits_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} rows to {full_path}")
    print(f"Saved {len(hits_df)} rows (explicit+implied) to {hits_path}")

    return full_path, hits_path


if __name__ == "__main__":
    from part4_main_loop import run_corpus

    results, failed_documents = run_corpus()

    rulebook_lookup = build_rulebook_lookup()
    df = enrich_results(results, rulebook_lookup)

    standard_rules = load_rulebook(RULEBOOK_PATH)
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    rule_index = build_rule_index(standard_rules, taxonomy)
    validate_coverage(df, rule_index)

    save_csv_outputs(df)

    if failed_documents:
        print(f"\n{len(failed_documents)} document(s) failed and are not included in the CSV outputs:")
        for f in failed_documents:
            print(f"  - {f}")
