"""
Week 6 — Qualitative Error Analysis sampling.

Produces two review sheets:

1. sample_hits_flagged.csv
   Every row from sap_iro_extraction_hits.csv (explicit + implied), with
   empty `error_type`/`reviewer_notes` columns added for manual annotation.

2. sample_notfound_priority.csv
   A priority subset of not_found rows from sap_iro_extraction_full.csv,
   ranked by semantic similarity between the rule's condition ("if..."
   clause + identification_cues) and the document text. High similarity but
   still not_found = plausible missed IRO -> review first for false
   negatives.

   Selection is threshold-based, not a fixed count: a row is included if its
   similarity score is at or above the SIMILARITY_PERCENTILE of the full
   not_found similarity distribution (data-driven, not an arbitrary cutoff).
   Documents with zero rows above that threshold still get a MIN_PER_DOC_FLOOR
   guarantee (their top-scoring rows), so no document goes unreviewed, and
   documents with many genuinely suspicious rows are not capped artificially.

Both files include empty `error_type` and `reviewer_notes` columns for
manual annotation.

Requires: the same corpus of source PDFs used by part4_main_loop.py, and an
OPENAI_API_KEY (used only for the not_found embedding step).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from company_config import get_company
from part1_taxonomy_mapping import load_rulebook
from part4_main_loop import CORPUS_DIR, extract_pdf_text

load_dotenv()
client = OpenAI()

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma
RULEBOOK_PATH = "IRO_rulebook_Final_1.json"
FULL_CSV_PATH = f"{COMPANY}_iro_extraction_full.csv"
HITS_CSV_PATH = f"{COMPANY}_iro_extraction_hits.csv"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100

SIMILARITY_PERCENTILE = 95    # global percentile of max_sim used as the review threshold
MIN_PER_DOC_FLOOR = 2         # guaranteed rows per document even if below threshold
MIN_CHUNK_CHARS = 40          # drop tiny/noise paragraphs when chunking doc_text
MAX_CHUNK_CHARS = 3000        # ~750 tokens; well under the 8192-token embedding limit

CONDITION_PATTERN = re.compile(r"^\s*If\s+(.*?)[,\u2014]\s*then\b", re.IGNORECASE | re.DOTALL)

OUTPUT_DIR = "."


# ---------------------------------------------------------------------------
# Rule condition text (strip the "if...then" scaffolding, keep the "if" part)
# ---------------------------------------------------------------------------

def extract_condition(rule_statement: str) -> str:
    match = CONDITION_PATTERN.match(rule_statement)
    return match.group(1).strip() if match else rule_statement.strip()


def build_rule_condition_lookup(rulebook_path: str = RULEBOOK_PATH) -> dict[str, str]:
    standard_rules = load_rulebook(rulebook_path)
    lookup = {}
    for rules in standard_rules.values():
        for rule in rules:
            condition = extract_condition(rule["rule_statement"])
            lookup[rule["rule_id"]] = f"{condition}. Relevant terms: {rule['identification_cues']}"
    return lookup


# ---------------------------------------------------------------------------
# Document text + chunking (cached per filename, since many rows share a doc)
# ---------------------------------------------------------------------------

def chunk_document(doc_text: str, min_chars: int = MIN_CHUNK_CHARS, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    raw_chunks = [c.strip() for c in re.split(r"\n\s*\n", doc_text) if c.strip()]

    chunks = []
    for raw in raw_chunks:
        if len(raw) <= max_chars:
            chunks.append(raw)
            continue
        # Paragraph too long (e.g. no blank-line breaks on this page) -> split
        # further on sentence boundaries so no single chunk risks exceeding
        # the embedding model's token limit.
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current.strip())

    return [c for c in chunks if len(c) >= min_chars]


class DocCache:
    """Caches extracted text + chunks per filename so each PDF is read once."""

    def __init__(self, corpus_dir: str = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self._chunks: dict[str, list[str]] = {}

    def _get_text(self, filename: str) -> str:
        path = Path(self.corpus_dir) / filename
        try:
            return extract_pdf_text(path)
        except Exception as e:
            print(f"[DocCache] failed to read {filename}: {e}")
            return ""

    def get_chunks(self, filename: str) -> list[str]:
        """Paragraph-level chunks, used for the not_found embedding-similarity check."""
        if filename not in self._chunks:
            self._chunks[filename] = chunk_document(self._get_text(filename))
        return self._chunks[filename]


# ---------------------------------------------------------------------------
# Embeddings (only used for the not_found / missed-IRO check)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536))
    # Defensive safety net: even after chunk_document's max_chars splitting,
    # truncate any stray oversized string (e.g. an unusually long rule
    # condition + cues) so a single bad input can't fail the whole batch.
    safe_texts = [t[:MAX_CHUNK_CHARS] if len(t) > MAX_CHUNK_CHARS else t for t in texts]
    vectors = []
    for i in range(0, len(safe_texts), EMBEDDING_BATCH_SIZE):
        batch = safe_texts[i : i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([d.embedding for d in response.data])
    return np.array(vectors)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# 1. Hits sample (full sample, no subsetting)
# ---------------------------------------------------------------------------

def build_hits_sample(hits_csv: str) -> pd.DataFrame:
    df = pd.read_csv(hits_csv)
    df["error_type"] = ""
    df["reviewer_notes"] = ""
    return df


# ---------------------------------------------------------------------------
# 2. not_found priority sample: rule condition vs doc chunks,
#    global percentile threshold + per-document minimum floor
# ---------------------------------------------------------------------------

def build_notfound_priority_sample(
    full_csv: str,
    rule_condition_lookup: dict[str, str],
    doc_cache: DocCache,
    percentile: float = SIMILARITY_PERCENTILE,
    min_per_doc_floor: int = MIN_PER_DOC_FLOOR,
) -> pd.DataFrame:
    df = pd.read_csv(full_csv)
    nf = df[df["evidence_status"] == "not_found"].copy()

    # Embed each unique rule condition once.
    unique_rule_ids = nf["rule_id"].unique().tolist()
    rule_texts = [rule_condition_lookup.get(rid, "") for rid in unique_rule_ids]
    rule_embeddings = embed_texts(rule_texts)
    rule_emb_lookup = dict(zip(unique_rule_ids, rule_embeddings))

    # Embed each unique document's chunks once.
    unique_filenames = nf["filename"].unique().tolist()
    doc_chunk_embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
    for filename in unique_filenames:
        chunks = doc_cache.get_chunks(filename)
        chunk_emb = embed_texts(chunks) if chunks else np.zeros((0, 1536))
        doc_chunk_embeddings[filename] = (chunks, chunk_emb)

    max_sims, best_previews = [], []
    for _, row in nf.iterrows():
        rule_emb = rule_emb_lookup.get(row["rule_id"])
        chunks, chunk_emb = doc_chunk_embeddings.get(row["filename"], ([], np.zeros((0, 1536))))
        if rule_emb is None or len(chunks) == 0:
            max_sims.append(0.0)
            best_previews.append("")
            continue
        sims = cosine_sim_matrix(rule_emb.reshape(1, -1), chunk_emb)[0]
        best_idx = int(np.argmax(sims))
        max_sims.append(round(float(sims[best_idx]), 3))
        best_previews.append(chunks[best_idx][:200])

    nf["flag_high_miss_similarity"] = max_sims
    nf["best_matching_chunk_preview"] = best_previews
    nf["error_type"] = ""
    nf["reviewer_notes"] = ""

    # Data-driven threshold: the Nth percentile of the *entire* not_found
    # similarity distribution, not a per-document cutoff. This lets rows
    # cluster wherever the real signal is, rather than forcing every
    # document to contribute the same fixed count.
    threshold = float(np.percentile(nf["flag_high_miss_similarity"], percentile))

    above_threshold = nf[nf["flag_high_miss_similarity"] >= threshold]

    # Floor: any document with zero rows above threshold still contributes
    # its top `min_per_doc_floor` rows, so no document goes completely
    # unreviewed even if it has no standout similarity scores.
    covered_filenames = set(above_threshold["filename"].unique())
    floor_rows = []
    for filename, group in nf.groupby("filename"):
        if filename in covered_filenames:
            continue
        floor_rows.append(group.sort_values("flag_high_miss_similarity", ascending=False).head(min_per_doc_floor))

    priority = pd.concat([above_threshold] + floor_rows, ignore_index=True) if floor_rows else above_threshold.copy()
    priority = priority.sort_values(["filename", "flag_high_miss_similarity"], ascending=[True, False]).reset_index(drop=True)

    print(f"[build_notfound_priority_sample] threshold (p{percentile}): {threshold:.3f}")
    print(f"[build_notfound_priority_sample] rows above threshold: {len(above_threshold)} "
          f"across {len(covered_filenames)} document(s)")
    if floor_rows:
        n_floor_docs = len(floor_rows)
        n_floor_rows = sum(len(g) for g in floor_rows)
        print(f"[build_notfound_priority_sample] floor applied to {n_floor_docs} document(s) "
              f"with no rows above threshold: +{n_floor_rows} row(s)")

    return priority


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    doc_cache = DocCache()

    print("Building hits sample (full, all rows)...")
    hits_df = build_hits_sample(HITS_CSV_PATH)
    hits_out = f"{OUTPUT_DIR}/{COMPANY}_sample_hits_flagged.csv"
    hits_df.to_csv(hits_out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(hits_df)} rows to {hits_out}")

    print("\nBuilding rule condition lookup...")
    rule_condition_lookup = build_rule_condition_lookup(RULEBOOK_PATH)

    print(f"\nBuilding not_found priority sample (embedding similarity, "
          f"p{SIMILARITY_PERCENTILE} threshold + floor of {MIN_PER_DOC_FLOOR}/doc)...")
    notfound_df = build_notfound_priority_sample(FULL_CSV_PATH, rule_condition_lookup, doc_cache)
    notfound_out = f"{OUTPUT_DIR}/{COMPANY}_sample_notfound_priority.csv"
    notfound_df.to_csv(notfound_out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(notfound_df)} rows to {notfound_out}")

    total_review = len(hits_df) + len(notfound_df)
    print(f"\nTotal rows for manual review: {total_review}")
