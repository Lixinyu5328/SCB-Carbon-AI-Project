"""
Week 6 — Part 7: wrong_tag match audit + IRO deduplication/merging.

Stage 0 — Match audit (revised)
    iro_type is not an independently-correctable field: it is fixed per
    rule_id in IRO_rulebook_Final_1.json (joined in at extraction time), and
    in this rulebook the overwhelming majority of esrs_sub_subtopic values
    have exactly one rule (avg ~1.12 rules/sub_subtopic) — there is usually
    no sibling rule with a different iro_type to redirect a row to. Checking
    reviewer_notes on the actual wrong_tag rows confirms this: the notes
    consistently say the evidence does not support the matched rule at all
    ("is not supported by the cited text", "is a mismatch"), not "this
    should have a different iro_type value."

    So for every row judged error_type == "wrong_tag", Stage 0 now asks an
    LLM-as-judge to confirm or overturn that flag — i.e. does the evidence
    actually support this rule_id, yes or no — using reviewer_notes as the
    authoritative signal when present (all 82 wrong_tag rows in the current
    data have non-empty, substantive reviewer_notes, so this is essentially
    formalizing the human reviewer's verdict; the rule_statement/excerpt are
    still passed as a cross-check for the rare case reviewer_notes is absent
    or ambiguous). Rows judged unsupported are excluded from the dedup
    register — they don't get relabeled to a different iro_type, they don't
    get embedded/clustered, they just don't count as a real IRO.

Stage A — Candidate generation (embedding)
    Within each iro_type partition (audited rows already excluded), embed
    esrs_topic + esrs_sub_subtopic + supporting_excerpt + reasoning with
    text-embedding-3-small and find permissive-threshold cosine-similarity
    pairs. esrs_topic is included alongside esrs_sub_subtopic because several
    sub_subtopic labels repeat verbatim across unrelated topics/standards
    (e.g. "Health and safety" appears under both S1 Own workforce and S4
    Consumers/end-users; "Reputation" appears under E2, E3, and E5) — without
    esrs_topic in the text, rows about completely different stakeholder
    groups could embed close together on the label alone.

Stage B — Cluster review (LLM)
    Each candidate cluster (not each pair — clusters are usually 2-4 rows,
    so this is far cheaper than O(n^2) pairwise calls) is sent to gpt-5-mini,
    which splits it into true-duplicate subgroups vs. merely-related rows,
    and for each duplicate subgroup names the most SPECIFIC/DETAILED member
    as the representative (extractive selection — no generated text). It is
    given esrs_topic alongside esrs_sub_subtopic for the same disambiguation
    reason as Stage A.

Outputs
    sap_iro_merge_mapping.csv     every original row + original_error_type,
                                  merge_group_id, dedup_role, related_iro_ids,
                                  dedup_rationale, dedup_confidence — full
                                  audit trail, one row per original
                                  extraction. dedup_role now also takes the
                                  value "excluded_unsupported" for rows
                                  Stage 0 disqualified; dedup_rationale holds
                                  the audit reason for those rows (reusing
                                  the same column merge groups use, rather
                                  than adding a new one).
    sap_iro_dedup_register.csv    same columns, but rows with
                                  dedup_role in ("merged", "excluded_unsupported")
                                  are dropped — this is the actual
                                  deduplicated, evidence-checked IRO list.

Requires OPENAI_API_KEY.
"""

import itertools
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel

from company_config import get_company

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma
INPUT_CSV_PATH = f"{COMPANY}_sample_hits_flagged.csv"
RULEBOOK_PATH = "IRO_rulebook_Final_1.json"

MERGE_MAPPING_PATH = f"{COMPANY}_iro_merge_mapping.csv"
DEDUP_REGISTER_PATH = f"{COMPANY}_iro_dedup_register.csv"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100

# Candidate generation is recall-first by design (Stage B does the precision
# work), so this is deliberately permissive rather than tuned for accuracy —
# but it should not be so loose that it's effectively no filter at all.
# Rationale for 0.80: text-embedding-3-small on short, jargon-dense domain
# text (shared ESRS/compliance vocabulary across rules) tends to sit in the
# 0.55-0.75 band for same-broad-topic-but-different-IRO pairs, and climbs
# above ~0.80 mainly when the underlying content (not just the vocabulary)
# actually overlaps. Combined with the esrs_topic addition above (which
# removes the main source of false-positive collisions from repeated
# sub_subtopic labels), 0.80 should catch true near-duplicates while keeping
# candidate clusters small enough for Stage B/human spot-checking to be
# practical. Still treat this as a knob: if a real run produces very few or
# very many candidate clusters, inspect the pairwise similarity distribution
# per iro_type partition and adjust.
CANDIDATE_THRESHOLD = 0.80

JUDGE_MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, ValueError)


# ---------------------------------------------------------------------------
# Stage 0: wrong_tag match audit (confirm or exclude, not relabel)
# ---------------------------------------------------------------------------

class MatchAudit(BaseModel):
    supports_rule: bool
    rationale: str


MATCH_AUDIT_SYSTEM_PROMPT = """You are auditing a single ESRS IRO (Impact, Risk, Opportunity) extraction that a separate review pass already flagged as "wrong_tag". iro_type is fixed by the matched rule_id in the rulebook (it is not a field you can reassign) — your only job is to decide whether the cited evidence actually supports THIS rule_id being triggered at all.

You are given: the rule_statement and identification_cues for the matched rule_id, the supporting_excerpt, the reasoning originally given for the match, and reviewer_notes (a human reviewer's own assessment of the problem, may be empty).

reviewer_notes, when non-empty, is authoritative and should be treated as the primary basis for your decision — if it states the evidence does not support this rule (e.g. the excerpt shows the opposite direction, a policy rather than an occurrence, or a related-but-distinct condition), follow that verdict directly rather than re-deriving your own reading of the excerpt. Only fall back to independent judgment from rule_statement + supporting_excerpt when reviewer_notes is empty or does not address whether the rule is supported.

Decide:
- supports_rule: false if the evidence does not actually satisfy rule_statement for this rule_id (this is the expected outcome for most wrong_tag rows — common patterns are policy-vs-actual-occurrence mismatches, e.g. a policy commitment cited as evidence of a negative impact, or evidence pointing the opposite direction from what the rule requires). true only if, on closer reading, the evidence does genuinely support this rule_id and the wrong_tag flag itself was a false alarm.
- rationale: one sentence justifying the decision, citing reviewer_notes directly if that is what the decision was based on.

Do not propose a different iro_type, rule_id, or any other field — that is out of scope here. This is a binary keep/exclude decision only."""


def load_rule_lookup(rulebook_path: str = RULEBOOK_PATH) -> dict[str, dict]:
    with open(rulebook_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    lookup = {}
    for rules in raw.values():
        for rule in rules:
            lookup[rule["rule_id"]] = rule
    return lookup


def audit_wrong_tag(row: pd.Series, rule_lookup: dict[str, dict]) -> MatchAudit | None:
    rule = rule_lookup.get(row["rule_id"], {})
    reviewer_notes = row.get("reviewer_notes", "")
    reviewer_notes = "" if pd.isna(reviewer_notes) else str(reviewer_notes)

    payload = {
        "rule_id": row["rule_id"],
        "rule_statement": rule.get("rule_statement", ""),
        "identification_cues": rule.get("identification_cues", ""),
        "iro_type": row["iro_type"],
        "supporting_excerpt": row["supporting_excerpt"],
        "reasoning": row["reasoning"],
        "reviewer_notes": reviewer_notes,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=JUDGE_MODEL,
                input=[
                    {"role": "system", "content": MATCH_AUDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=MatchAudit,
            )
            if response.output_parsed is None:
                raise ValueError("model returned no parsed output (possible refusal)")
            return response.output_parsed
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"[audit_wrong_tag] {row['rule_id']} attempt {attempt} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)

    print(f"[audit_wrong_tag] failed after {MAX_RETRIES} attempts for rule_id={row['rule_id']}: {last_error}")
    return None


def apply_wrong_tag_audit(df: pd.DataFrame) -> pd.DataFrame:
    wrong_tag_idx = df.index[df["error_type"] == "wrong_tag"]
    if len(wrong_tag_idx) == 0:
        print("No wrong_tag rows found; skipping match audit stage.")
        return df

    rule_lookup = load_rule_lookup()
    excluded_count = 0

    for idx in wrong_tag_idx:
        audit = audit_wrong_tag(df.loc[idx], rule_lookup)
        if audit is None:
            continue
        if not audit.supports_rule:
            df.at[idx, "dedup_role"] = "excluded_unsupported"
            df.at[idx, "dedup_rationale"] = audit.rationale
            excluded_count += 1
            print(f"  excluded rule_id={df.at[idx, 'rule_id']}: {audit.rationale}")
        else:
            print(f"  kept rule_id={df.at[idx, 'rule_id']} (wrong_tag flag overturned): {audit.rationale}")

    print(f"Stage 0: excluded {excluded_count} of {len(wrong_tag_idx)} wrong_tag rows as unsupported matches.")
    return df


# ---------------------------------------------------------------------------
# Stage A: embedding-based candidate clustering
# ---------------------------------------------------------------------------

def build_embedding_text(row: pd.Series) -> str:
    return f"{row['esrs_topic']} / {row['esrs_sub_subtopic']}. {row['supporting_excerpt']} {row['reasoning']}"


def get_embeddings_batched(texts: list[str]) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
    return np.array(all_embeddings)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return norm @ norm.T


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_candidate_clusters(partition: pd.DataFrame, embeddings: np.ndarray) -> list[list[str]]:
    ids = partition["iro_uid"].tolist()
    uf = UnionFind(ids)
    sim = cosine_similarity_matrix(embeddings)

    for i, j in itertools.combinations(range(len(ids)), 2):
        if sim[i, j] >= CANDIDATE_THRESHOLD:
            uf.union(ids[i], ids[j])

    clusters = defaultdict(list)
    for i in ids:
        clusters[uf.find(i)].append(i)

    return [members for members in clusters.values() if len(members) >= 2]


# ---------------------------------------------------------------------------
# Stage B: LLM cluster review
# ---------------------------------------------------------------------------

class DuplicateGroup(BaseModel):
    member_ids: list[str]
    representative_id: str
    rationale: str
    confidence: float


class ClusterReview(BaseModel):
    duplicate_groups: list[DuplicateGroup]
    related_ids: list[str]


CLUSTER_SYSTEM_PROMPT = """You are reviewing a small cluster of candidate-duplicate IRO (Impact, Risk, Opportunity) records extracted from ESRS sustainability disclosures. All records in this cluster share the same iro_type and were flagged as textually similar by an embedding model, but embedding similarity does not mean they are true duplicates — you must judge based on meaning. Note that some esrs_sub_subtopic labels (e.g. "Health and safety", "Reputation") repeat verbatim across completely unrelated esrs_topic areas, so always read esrs_topic alongside esrs_sub_subtopic rather than treating a matching sub_subtopic label as evidence of similarity on its own.

For each record you are given: member_id, esrs_topic, esrs_sub_subtopic, supporting_excerpt, reasoning.

Your task:
1. Identify which records describe the SAME underlying impact/risk/opportunity (true duplicates) — even if worded differently, extracted from different documents, or matched to a different rule_id — and group them into duplicate_groups.
2. A record only belongs in a duplicate_group with others if it is truly the same underlying IRO, not merely the same topic/label or the same evidence passage supporting a different angle. If two records share evidence, topic, or sub_subtopic label but describe a materially different underlying IRO, keep them out of duplicate_groups and list their ids in related_ids instead.
3. Within each duplicate_group, choose representative_id: the member whose supporting_excerpt and reasoning together describe the IRO most SPECIFICALLY and concretely (most detail on mechanism, location, affected stakeholders, or quantification) — not the longest text, and not any confidence score, but the most specific and informative description of the underlying IRO.
4. Give a one-sentence rationale per duplicate_group explaining why its members are the same IRO and why that representative was chosen over the others.
5. Give a confidence score (0-1) per duplicate_group.
6. A record with no true duplicate in this cluster but that is topically related to one or more others should be listed once in related_ids, not placed in any duplicate_group.
7. A record with no true duplicate and no meaningful relation to anything else should not appear anywhere in your output — do not force every member into a group.

Return only the structured result; do not invent member_ids that were not given to you."""


def review_cluster(df: pd.DataFrame, member_ids: list[str]) -> ClusterReview | None:
    members = df[df["iro_uid"].isin(member_ids)]
    payload = members[["iro_uid", "esrs_topic", "esrs_sub_subtopic", "supporting_excerpt", "reasoning"]].rename(
        columns={"iro_uid": "member_id"}
    ).to_dict(orient="records")

    user_prompt = f"Cluster members ({len(payload)} total):\n{json.dumps(payload, ensure_ascii=False)}"
    valid_ids = set(member_ids)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=JUDGE_MODEL,
                input=[
                    {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=ClusterReview,
            )
            if response.output_parsed is None:
                raise ValueError("model returned no parsed output (possible refusal)")

            parsed = response.output_parsed
            cleaned_groups = []
            for g in parsed.duplicate_groups:
                g.member_ids = [m for m in g.member_ids if m in valid_ids]
                if len(g.member_ids) >= 2 and g.representative_id in valid_ids:
                    cleaned_groups.append(g)
            parsed.duplicate_groups = cleaned_groups
            parsed.related_ids = [i for i in parsed.related_ids if i in valid_ids]
            return parsed
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"[review_cluster] attempt {attempt} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)

    print(f"[review_cluster] failed after {MAX_RETRIES} attempts for cluster {member_ids}: {last_error}")
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_dedup(input_path: str = INPUT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # provenance: freeze the judged error_type before Stage 0 adds dedup_role/rationale
    df["original_error_type"] = df["error_type"]

    df = df.reset_index(drop=True)
    df["iro_uid"] = df["doc_id"].astype(str) + "_" + df["rule_id"].astype(str) + "_" + df.index.astype(str)

    df["merge_group_id"] = pd.NA
    df["dedup_role"] = "unique"        # unique | representative | merged | related | excluded_unsupported
    df["related_iro_ids"] = ""
    df["dedup_rationale"] = ""
    df["dedup_confidence"] = pd.NA

    df = apply_wrong_tag_audit(df)

    group_counter = 0
    total_candidate_clusters = 0

    # excluded rows never enter embedding/clustering — they aren't real IROs
    clusterable = df[df["dedup_role"] != "excluded_unsupported"]

    for iro_type, partition in clusterable.groupby("iro_type"):
        if len(partition) < 2:
            continue

        texts = partition.apply(build_embedding_text, axis=1).tolist()
        embeddings = get_embeddings_batched(texts)
        clusters = find_candidate_clusters(partition, embeddings)
        total_candidate_clusters += len(clusters)
        print(f"[{iro_type}] {len(partition)} rows -> {len(clusters)} candidate cluster(s)")

        for member_ids in clusters:
            review = review_cluster(df, member_ids)
            if review is None:
                continue

            for dgroup in review.duplicate_groups:
                group_counter += 1
                group_id = f"grp_{group_counter:04d}"

                mask = df["iro_uid"].isin(dgroup.member_ids)
                df.loc[mask, "merge_group_id"] = group_id
                df.loc[mask, "dedup_role"] = "merged"
                df.loc[mask, "dedup_rationale"] = dgroup.rationale
                df.loc[mask, "dedup_confidence"] = dgroup.confidence
                df.loc[df["iro_uid"] == dgroup.representative_id, "dedup_role"] = "representative"

            for rid in review.related_ids:
                peers = [x for x in review.related_ids if x != rid]
                row_mask = df["iro_uid"] == rid
                if (df.loc[row_mask, "dedup_role"] == "unique").all():
                    df.loc[row_mask, "dedup_role"] = "related"
                df.loc[row_mask, "related_iro_ids"] = ",".join(peers)

    print(f"\nCandidate clusters from embedding stage: {total_candidate_clusters}")
    print(f"Confirmed duplicate groups after LLM review: {group_counter}")

    df.to_csv(MERGE_MAPPING_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved full mapping ({len(df)} rows, every original extraction) to {MERGE_MAPPING_PATH}")

    register = df[~df["dedup_role"].isin(["merged", "excluded_unsupported"])].copy()
    register.to_csv(DEDUP_REGISTER_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved deduplicated register ({len(register)} rows) to {DEDUP_REGISTER_PATH}")

    return df


if __name__ == "__main__":
    run_dedup()
