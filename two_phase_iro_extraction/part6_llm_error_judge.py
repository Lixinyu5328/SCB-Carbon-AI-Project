"""
Week 6 - Automated error_type judging pipeline.

Replaces manual row-by-row review with an LLM judge that re-checks each
extracted IRO (or each not_found candidate) against the actual source
document text, then fills in error_type / reviewer_notes automatically.


What it does:
1. hits rows  -> LLM judge decides: correct / fabricated / wrong_tag /
   too_vague / other. Uses the FULL text of the source document (not
   retrieved chunks) so a fabrication/tag check can never miss evidence
   just because embedding retrieval didn't surface the right paragraph.
   Documents here are short policy PDFs, so this comfortably fits in one
   call.
2. notfound rows -> LLM judge decides: missed_iro / correct_notfound /
   other, using the rule condition + top-k retrieved chunks (kept as
   retrieval rather than full text, since here the judge has to be re-run
   per rule per document, and full-text-per-rule would multiply cost with
   little benefit -- see NOTFOUND_TOP_K below).

Duplicate detection is intentionally NOT handled here -- it requires
comparing rows to each other rather than to the source doc, and is being
done as a separate pass after this error-type pass.

Resumable: every row is written to the CSV immediately after it's judged, and
any row whose error_type is already non-empty is skipped on the next run. So
if it hangs or you Ctrl+C it, everything up to that point is safely saved --
just re-run the same command to continue from where it stopped.

Requires: OPENAI_API_KEY, the same corpus of source PDFs used elsewhere in
the pipeline, and the same files as part6_error_analysis_sampling.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from company_config import get_company
from part1_taxonomy_mapping import load_rulebook
from part6_error_analysis_sampling import (
    DocCache, build_rule_condition_lookup, embed_texts, cosine_sim_matrix,
    RULEBOOK_PATH,
)

load_dotenv()
client = OpenAI()

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma
HITS_PATH = f"{COMPANY}_sample_hits_flagged.csv"
NOTFOUND_PATH = f"{COMPANY}_sample_notfound_priority.csv"

JUDGE_MODEL = "gpt-5-mini"      # current cost-efficient tier as of mid-2026; swap for gpt-5.5 if budget allows
NOTFOUND_TOP_K = 6                # notfound only: retrieval can miss the decisive chunk at k=3
SPOT_CHECK_FRACTION = 0.10        # fraction of auto-judged rows to sample for manual calibration

HITS_LABELS = ["correct", "fabricated", "wrong_tag", "too_vague", "other"]
NOTFOUND_LABELS = ["missed_iro", "correct_notfound", "other"]


# ---------------------------------------------------------------------------
# Rule text lookup (full statement, not just the "if" condition, since the
# judge needs the full rule to check tag correctness too)
# ---------------------------------------------------------------------------

def build_full_rule_lookup(rulebook_path: str = RULEBOOK_PATH) -> dict[str, dict]:
    standard_rules = load_rulebook(rulebook_path)
    lookup = {}
    for rules in standard_rules.values():
        for rule in rules:
            lookup[rule["rule_id"]] = rule
    return lookup


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def build_full_text_lookup(filenames: list[str], doc_cache: DocCache) -> dict[str, str]:
    """hits judge gets the whole document rather than retrieved chunks --
    these policy PDFs are short, so there's no reason to risk retrieval
    missing the decisive paragraph. DocCache has no public full-text getter,
    so this rejoins its already-computed paragraph chunks (each >=40 chars,
    per part6_error_analysis_sampling's MIN_CHUNK_CHARS -- a few very short
    stray paragraphs may be dropped, which is immaterial for this purpose)."""
    lookup = {}
    for filename in set(filenames):
        lookup[filename] = "\n\n".join(doc_cache.get_chunks(filename))
    return lookup


def top_k_chunks(query: str, chunks: list[str], k: int = NOTFOUND_TOP_K) -> list[str]:
    """notfound only: still uses embedding retrieval (re-running per rule
    against full document text would multiply cost for little benefit,
    since notfound rows are evaluated one rule at a time)."""
    if not chunks or not query.strip():
        return []
    query_emb = embed_texts([query])
    chunk_emb = embed_texts(chunks)
    sims = cosine_sim_matrix(query_emb, chunk_emb)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_idx]


# ---------------------------------------------------------------------------
# LLM judge calls
# ---------------------------------------------------------------------------

def call_judge(system_prompt: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        print(f"[call_judge] failed to parse response: {e}")
        return {"error_type": "other", "confidence": 0.0, "notes": "judge response parse failure"}


HITS_SYSTEM_PROMPT = f"""You are auditing an automated ESRS double-materiality IRO extraction pipeline.
For each extracted IRO, decide the single best error_type from: {HITS_LABELS}.
- "correct": the source text genuinely supports this IRO, and the ESRS standard/topic/iro_type tag fits the rule.
- "fabricated": the supporting_excerpt is not actually present/supported in the retrieved source chunks.
- "wrong_tag": the underlying evidence is real, but this rule/standard/iro_type is the wrong fit for it.
- "too_vague": the evidence is real and correctly tagged, but too generic/boilerplate to be an actionable IRO.
- "other": none of the above fit; explain in notes.
Respond with ONLY a JSON object: {{"error_type": "...", "confidence": 0.0-1.0, "notes": "one or two sentences"}}"""

NOTFOUND_SYSTEM_PROMPT = f"""You are auditing an automated ESRS double-materiality IRO extraction pipeline for
false negatives. A rule was marked not_found for this document, but retrieval flagged these chunks as
semantically similar to the rule's condition. Decide from: {NOTFOUND_LABELS}.
- "missed_iro": the retrieved chunks actually do satisfy the rule's condition -- this is a genuine miss.
- "correct_notfound": the chunks are topically similar but don't actually satisfy the rule's condition.
- "other": ambiguous or insufficient information; explain in notes.
Respond with ONLY a JSON object: {{"error_type": "...", "confidence": 0.0-1.0, "notes": "one or two sentences"}}"""


def judge_hit_row(row: pd.Series, rule: dict, full_doc_text: str) -> dict:
    user_prompt = f"""Rule ({row['rule_id']}): {rule.get('rule_statement', 'N/A')}
Identification cues: {rule.get('identification_cues', 'N/A')}
Assigned ESRS: standard={row.get('standard')}, topic={row.get('esrs_topic')}, subtopic={row.get('esrs_subtopic')}, sub_subtopic={row.get('esrs_sub_subtopic')}
iro_type: {row.get('iro_type')}
evidence_status: {row.get('evidence_status')}
Model's supporting_excerpt: {row.get('supporting_excerpt')}
Model's reasoning: {row.get('reasoning')}

Full source document text (verify the excerpt against this directly):
{full_doc_text if full_doc_text.strip() else "(document text unavailable)"}"""
    return call_judge(HITS_SYSTEM_PROMPT, user_prompt)


def judge_notfound_row(row: pd.Series, rule: dict, chunks: list[str]) -> dict:
    user_prompt = f"""Rule ({row['rule_id']}): {rule.get('rule_statement', 'N/A')}
Identification cues: {rule.get('identification_cues', 'N/A')}
iro_type: {row.get('iro_type')}
Assigned ESRS: standard={row.get('standard')}, topic={row.get('esrs_topic')}, sub_subtopic={row.get('esrs_sub_subtopic')}

Top {len(chunks)} retrieved source chunks (highest similarity to the rule condition):
{chr(10).join(f"[{i+1}] {c}" for i, c in enumerate(chunks)) if chunks else "(no chunks retrieved)"}"""
    return call_judge(NOTFOUND_SYSTEM_PROMPT, user_prompt)


# ---------------------------------------------------------------------------
# Main per-file pipelines
# ---------------------------------------------------------------------------

def _ensure_judge_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "error_type" not in df.columns:
        df["error_type"] = ""
    if "auto_confidence" not in df.columns:
        df["auto_confidence"] = np.nan
    if "reviewer_notes" not in df.columns:
        df["reviewer_notes"] = ""
    df["error_type"] = df["error_type"].fillna("").astype(str)
    df["reviewer_notes"] = df["reviewer_notes"].fillna("").astype(str)
    return df


def run_hits_pipeline(path: str, doc_cache: DocCache, rule_lookup: dict[str, dict]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    df = _ensure_judge_columns(df)

    pending = df[df["error_type"].str.strip() == ""].index.tolist()
    print(f"{len(df) - len(pending)}/{len(df)} rows already judged, {len(pending)} remaining.")
    if not pending:
        return df

    full_text_lookup = build_full_text_lookup(df.loc[pending, "filename"].tolist(), doc_cache)

    try:
        for i, idx in enumerate(pending):
            row = df.loc[idx]
            rule = rule_lookup.get(row["rule_id"], {})
            full_doc_text = full_text_lookup.get(row["filename"], "")
            result = judge_hit_row(row, rule, full_doc_text)

            df.at[idx, "error_type"] = result.get("error_type", "other")
            df.at[idx, "auto_confidence"] = result.get("confidence", 0.0)
            df.at[idx, "reviewer_notes"] = result.get("notes", "")
            df.to_csv(path, index=False, encoding="utf-8-sig")  # save after every row

            if (i + 1) % 25 == 0 or (i + 1) == len(pending):
                print(f"  ...{i + 1}/{len(pending)} judged and saved")
    except KeyboardInterrupt:
        print(f"\nInterrupted. Progress up to this row is already saved to {path} -- "
              f"re-run the same command to pick up where you left off.")
    return df


def run_notfound_pipeline(path: str, doc_cache: DocCache, rule_condition_lookup: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    df = _ensure_judge_columns(df)

    pending = df[df["error_type"].str.strip() == ""].index.tolist()
    print(f"{len(df) - len(pending)}/{len(df)} rows already judged, {len(pending)} remaining.")
    if not pending:
        return df

    try:
        for i, idx in enumerate(pending):
            row = df.loc[idx]
            rule_text = rule_condition_lookup.get(row["rule_id"], "")
            chunks = doc_cache.get_chunks(row["filename"])
            top_chunks = top_k_chunks(rule_text, chunks)
            result = judge_notfound_row(row, {"rule_statement": rule_text}, top_chunks)

            df.at[idx, "error_type"] = result.get("error_type", "other")
            df.at[idx, "auto_confidence"] = result.get("confidence", 0.0)
            df.at[idx, "reviewer_notes"] = result.get("notes", "")
            df.to_csv(path, index=False, encoding="utf-8-sig")  # save after every row

            if (i + 1) % 25 == 0 or (i + 1) == len(pending):
                print(f"  ...{i + 1}/{len(pending)} judged and saved")
    except KeyboardInterrupt:
        print(f"\nInterrupted. Progress up to this row is already saved to {path} -- "
              f"re-run the same command to pick up where you left off.")
    return df


def write_spot_check_sample(df: pd.DataFrame, source_name: str, fraction: float = SPOT_CHECK_FRACTION) -> pd.DataFrame:
    """Stratified sample by error_type, biased toward low-confidence rows,
    for a quick manual calibration check (NOT the full manual review)."""
    n = max(1, int(len(df) * fraction))
    df = df.copy()
    df["_source"] = source_name
    sampled = (
        df.sort_values("auto_confidence")
        .groupby("error_type", group_keys=False)
        .apply(lambda g: g.head(max(1, int(len(g) * fraction))))
    )
    return sampled.head(n).drop(columns=["_source"], errors="ignore")


if __name__ == "__main__":
    doc_cache = DocCache()
    rule_lookup = build_full_rule_lookup(RULEBOOK_PATH)
    rule_condition_lookup = build_rule_condition_lookup(RULEBOOK_PATH)

    print("Running LLM judge on hits sample (saves after every row, resumable if interrupted)...")
    hits_df = run_hits_pipeline(HITS_PATH, doc_cache, rule_lookup)
    print(f"  error_type counts:\n{hits_df['error_type'].value_counts()}")

    print("\nRunning LLM judge on notfound sample (saves after every row, resumable if interrupted)...")
    notfound_df = run_notfound_pipeline(NOTFOUND_PATH, doc_cache, rule_condition_lookup)
    print(f"  error_type counts:\n{notfound_df['error_type'].value_counts()}")

    print("\nBuilding stratified spot-check sample for manual calibration...")
    spot_check = pd.concat([
        write_spot_check_sample(hits_df, "hits"),
        write_spot_check_sample(notfound_df, "notfound"),
    ], ignore_index=True)
    spot_check_path = f"{COMPANY}_spot_check_sample.csv"
    spot_check.to_csv(spot_check_path, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(spot_check)} rows to {spot_check_path} -- "
          f"review these manually (e.g. with part6_annotate_errors.py --path {spot_check_path}) "
          f"to estimate how reliable the automated labels are before trusting the rest.")
