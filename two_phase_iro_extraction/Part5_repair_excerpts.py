import re
import shutil
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from company_config import get_company
from part3_extract import MAX_RETRIES, MODEL, RETRY_BACKOFF_SECONDS, RETRYABLE_ERRORS, client
from part4_main_loop import CORPUS_DIR, extract_pdf_text
from part5_output import HIT_STATUSES

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma
FULL_CSV = f"{COMPANY}_iro_extraction_full.csv"
HITS_CSV = f"{COMPANY}_iro_extraction_hits.csv"
QUOTE_CHARS = ('"', "\u201c", "\u201d")

REPAIR_SYSTEM_PROMPT = """You are given the full text of a company document and a list of rules that were previously matched to it, but with paraphrased evidence instead of a direct quote.

For each rule, find the shortest verbatim passage in the document (max 40 words) that best supports it. Copy it EXACTLY as written in the document, including punctuation and capitalization. Do not paraphrase, summarize, or reference section numbers instead of quoting.

You must return exactly one result per rule_id provided, using the same rule_id values."""


class ExcerptRepairItem(BaseModel):
    rule_id: str
    quote: str


class ExcerptRepairBatch(BaseModel):
    repairs: list[ExcerptRepairItem]


def has_any_quote(value: str) -> bool:
    text = str(value)
    return any(ch in text for ch in QUOTE_CHARS)


def build_repair_prompt(doc_text: str, items: list[dict]) -> str:
    lines = [
        f'- rule_id: {i["rule_id"]}\n  reasoning: {i["reasoning"]}\n  old_paraphrase: {i["old_excerpt"]}'
        for i in items
    ]
    return (
        f'Document content:\n"""\n{doc_text}\n"""\n\n'
        f"Rules needing a verbatim quote ({len(items)} total):\n" + "\n".join(lines)
    )


def repair_file(filename: str, doc_text: str, items: list[dict]) -> dict[str, str]:
    prompt = build_repair_prompt(doc_text, items)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text_format=ExcerptRepairBatch,
            )
            if response.output_parsed is None:
                raise ValueError("model returned no parsed output")
            return {r.rule_id: r.quote for r in response.output_parsed.repairs}
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"[repair_file] {filename} attempt {attempt} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)

    print(f"[repair_file] {filename} failed after {MAX_RETRIES} attempts: {last_error}")
    return {}


def main():
    shutil.copy(FULL_CSV, FULL_CSV.replace(".csv", "_backup.csv"))
    print(f"Backed up {FULL_CSV} -> {FULL_CSV.replace('.csv', '_backup.csv')}")

    df = pd.read_csv(FULL_CSV)
    needs_repair = df["evidence_status"].isin(["explicit", "implied"]) & ~df["supporting_excerpt"].apply(has_any_quote)
    print(f"{needs_repair.sum()} rows need repair across {df.loc[needs_repair, 'filename'].nunique()} file(s)")

    # Preserve the pre-repair text for every row (not just repaired ones) so the
    # original paraphrase/evidence is never lost even if this script is re-run
    # later and the single backup file gets overwritten in the process.
    df["original_supporting_excerpt"] = df["supporting_excerpt"]

    df["excerpt_verified"] = pd.NA

    for filename, group in df[needs_repair].groupby("filename"):
        doc_text = extract_pdf_text(Path(CORPUS_DIR) / filename)
        items = [
            {"rule_id": row["rule_id"], "reasoning": row["reasoning"], "old_excerpt": row["supporting_excerpt"]}
            for _, row in group.iterrows()
        ]
        quotes = repair_file(filename, doc_text, items)

        normalized_doc = re.sub(r"\s+", " ", doc_text)
        for idx, row in group.iterrows():
            quote = quotes.get(row["rule_id"])
            if quote is None:
                print(f"[repair] {filename} {row['rule_id']}: no repair returned, left unchanged")
                continue
            quote = quote.strip().strip("\"\u201c\u201d")
            verified = re.sub(r"\s+", " ", quote) in normalized_doc
            df.at[idx, "supporting_excerpt"] = f'"{quote}"'
            df.at[idx, "excerpt_verified"] = verified

        print(f"[repair] {filename}: repaired {len(quotes)}/{len(items)} rows")

    n_unverified = (df["excerpt_verified"] == False).sum()  # noqa: E712
    print(f"\nDone. {n_unverified} repaired row(s) could not be verified as exact substrings of the source PDF text — check these manually.")

    df.to_csv(FULL_CSV, index=False, encoding="utf-8-sig")
    hits_df = df[df["evidence_status"].isin(HIT_STATUSES)]
    hits_df.to_csv(HITS_CSV, index=False, encoding="utf-8-sig")
    print(f"Updated {FULL_CSV} ({len(df)} rows) and {HITS_CSV} ({len(hits_df)} rows)")


if __name__ == "__main__":
    main()
