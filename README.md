# DMA Pipeline — Usage Guide

The pipeline runs in two stages: **1. Two-Phase IRO Extraction** (produces a
deduplicated IRO register for a company) and **2. DMA Scoring** (scores each
IRO and filters material subtopics/IROs). Run stage 1 to completion before
starting stage 2.

Every script accepts a `--company <name>` flag (e.g. `--company sap`,
`--company puma`). It sets the default input/output file names and the
document corpus directory for that run; omit it and it defaults to `sap`.
Any individual path still has its own override flag where the script
exposes one (see each step below).

**Requirements**: `OPENAI_API_KEY` set in the environment or a `.env` file;
`pip install openai pandas pdfplumber python-dotenv pydantic python-docx numpy`.

---

## Stage 1 — `two_phase_iro_extraction/`

### Library modules — not run directly
`part1_taxonomy_mapping.py`, `part2_schemas.py`, `part3_extract.py`, and
`part4_main_loop.py` are imported by the two commands below; you don't
invoke them yourself. (`part4_main_loop.py` does have its own `__main__`
block, but it only re-runs the same extraction with no output-saving step —
use `part5_output.py` instead, which does the same extraction and saves
the CSVs.)

### Commands, in order

**1. `python part5_output.py --company <name>`**
Runs the full extraction: loads `IRO_rulebook_Final_1.json` +
`Document_taxonomy_1.docx`, processes every PDF in the company's document
corpus (one call per document to gpt-5-mini per rule set), joins each hit
back to its rulebook fields, and validates that every document produced the
expected number of rule rows.
- Output: `{company}_iro_extraction_full.csv` (every rule × document row)
  and `{company}_iro_extraction_hits.csv` (explicit/implied rows only).

**2. `python Part5_repair_excerpts.py --company <name>`**
Reads `{company}_iro_extraction_full.csv`. For any hit whose
`supporting_excerpt` isn't an actual quoted string, asks gpt-5-mini for a
verbatim ≤40-word quote from the source PDF and verifies it's an exact
substring.
- Backs up the full CSV first (`_backup.csv`).
- Re-saves `{company}_iro_extraction_full.csv` (adds `original_supporting_excerpt`,
  `excerpt_verified`) and regenerates `{company}_iro_extraction_hits.csv`.

**3. `python part6_error_analysis_sampling.py --company <name>`**
Reads the full/hits CSVs from steps 1–2.
- Output: `{company}_sample_hits_flagged.csv` (every hit row + empty
  `error_type`/`reviewer_notes` for annotation) and
  `{company}_sample_notfound_priority.csv` (a data-driven priority subset
  of `not_found` rows, ranked by embedding similarity between the rule
  condition and the document text — text-embedding-3-small — to surface
  likely missed IROs; threshold = 95th percentile with a 2-row-per-document
  floor).

**4. `python part6_llm_error_judge.py --company <name>`**
Reads and rewrites the two sample CSVs from step 3 in place (resumable —
safe to interrupt and re-run). Uses gpt-5-mini as a judge to fill in
`error_type` / `auto_confidence` / `reviewer_notes` for every row:
- Hits: `correct` / `fabricated` / `wrong_tag` / `too_vague` / `other`
  (checked against the full source-document text).
- Not-found: `missed_iro` / `correct_notfound` / `other` (checked against
  the top-6 retrieved chunks).
- Also writes `{company}_spot_check_sample.csv`, a small stratified sample
  for manual calibration of the automated judgments.

**5. `python part7_dedup.py --company <name>`**
Reads `{company}_sample_hits_flagged.csv` (now judged by step 4).
- Stage 0: gpt-5-mini re-audits every `wrong_tag` row and excludes
  unsupported matches.
- Stage A: embeds remaining rows (text-embedding-3-small) within each
  `iro_type` and clusters candidates at cosine similarity ≥ 0.80.
- Stage B: gpt-5-mini reviews each candidate cluster, splitting it into
  true-duplicate groups (with one chosen representative) vs. merely-related
  rows.
- Output: `{company}_iro_merge_mapping.csv` (every original row + full
  audit trail) and `{company}_iro_dedup_register.csv` (the deduplicated
  IRO list — this is the input to Stage 2).

---

## Stage 2 — `DMA_scoring/`

**1. `python part1_secondary_dma_extraction.py --company <name>`**
Reads `{company}_iro_dedup_register.csv`, re-opens each IRO's source
document, and extracts (not scores) the evidence needed for DMA scoring:
IRO status, time horizon, scale/scope/irremediability (impacts), likelihood,
and supported financial-effect channels (risks/opportunities). gpt-5-mini,
resumable.
- Output: `{company}_dma_secondary_extraction.jsonl` + `.csv`, and
  `{company}_dma_secondary_failures.csv` for rows that failed after retries.
- Useful flags: `--max-rows N` (small test run), `--overwrite`,
  `--print-schema` (print the structured-output schema and exit).

**2. `python part2_enrichment_representative.py --company <name>`**
Reads the baseline JSONL from step 1, `{company}_iro_dedup_register.csv`,
and `{company}_iro_merge_mapping.csv`. For every `representative` IRO (one
that absorbed duplicates during dedup), gathers evidence from all merged
members and asks gpt-5-mini to update DMA fields where the combined
evidence is more direct/specific/complete. `unique`/`related` IROs pass
through unchanged; merged members never get their own score.
- Output: `{company}_dma_secondary_extraction_enriched.jsonl` + `.csv`,
  plus `{company}_dma_representative_enrichment_audit.csv` and a failures CSV.
- Useful flags: `--max-representatives N`, `--overwrite`, `--print-schema`.

**3. `python part3_dma_scoring.py --company <name>`**
Reads the enriched JSONL + dedup register from steps 1–2. Scores each
applicable DMA dimension 1–5 with gpt-5-mini against the fixed DMA Plan
criteria; uses cautious `assumption_based` scoring when a field is
`not_disclosed`.
- Output: `{company}_dma_dimension_scores.jsonl` + `.csv`.
- Useful flags: `--limit N`, `--overwrite`, `--iro-type {negative_impact,
  positive_impact,risk,opportunity}` (score only one category),
  `--replace-selected` (rescore only `--iro-type`, keep the rest).

**4. `python Part4_filter_material_subtopics_and_iros.py --company <name>`**
Reads `{company}_dma_dimension_scores.jsonl` +
`{company}_dma_secondary_extraction_enriched.csv`. Applies the materiality
rule: an IRO is material if `final_score >= threshold` (default 15); a
subtopic is material if it contains at least one material IRO.
- Output: `material_subtopics.csv` and `material_iros.csv` (not
  company-prefixed — these two final files use a fixed name regardless of
  which company was run).
- Useful flags: `--output-dir DIR`, `--threshold N`.

---

## End-to-end example (one company)

```bash
cd two_phase_iro_extraction
python part5_output.py --company puma
python Part5_repair_excerpts.py --company puma
python part6_error_analysis_sampling.py --company puma
python part6_llm_error_judge.py --company puma
python part7_dedup.py --company puma

cd ../DMA_scoring
python part1_secondary_dma_extraction.py --company puma
python part2_enrichment_representative.py --company puma
python part3_dma_scoring.py --company puma
python Part4_filter_material_subtopics_and_iros.py --company puma
```

## Corpus directory path convention

The company-document corpus directory is not a literal path in the code —
it's generated by `corpus_dir_for()` in `company_config.py`
(`two_phase_iro_extraction/company_config.py` and
`DMA_scoring/company_config.py`), and used by `part4_main_loop.py`,
`part1_secondary_dma_extraction.py`, and `part2_enrichment_representative.py`:

```python
DOCUMENTS_ROOT = "/Users/lixinyu/Documents/CarbonAI_code"

def corpus_dir_for(company: str) -> str:
    tag = company.upper()
    return f"{DOCUMENTS_ROOT}/{tag}_IROs_extraction/{tag}_documents"
```

For `--company puma`, this resolves to:
/Users/lixinyu/Documents/CarbonAI_code/PUMA_IROs_extraction/PUMA_documents

If a company's real folder doesn't match this pattern, either:
1. edit `DOCUMENTS_ROOT` in both `company_config.py` files to your actual
   root, keeping the `{TAG}_IROs_extraction/{TAG}_documents` folder naming, or
2. pass `--corpus-dir /actual/full/path` explicitly to bypass auto-derivation
   for that run.