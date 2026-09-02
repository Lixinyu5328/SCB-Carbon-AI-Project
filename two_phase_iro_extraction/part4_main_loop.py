import time
from pathlib import Path

import pdfplumber

from company_config import corpus_dir_for, get_company
from part1_taxonomy_mapping import build_rule_index, load_rulebook, load_taxonomy, merge_rule_entries, parse_doc_ids
from part2_schemas import build_rule_inputs
from part3_extract import IROExtractionError, extract_iros

RULEBOOK_PATH = "IRO_rulebook_Final_1.json"
TAXONOMY_PATH = "Document_taxonomy_1.docx"
COMPANY = get_company()  # set via --company, e.g. --company sap / --company puma
CORPUS_DIR = corpus_dir_for(COMPANY)


def extract_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages).strip()


def load_pipeline_context() -> dict[int, dict]:
    standard_rules = load_rulebook(RULEBOOK_PATH)
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    return build_rule_index(standard_rules, taxonomy)


def process_document(path: Path, rule_index: dict[int, dict]) -> list[dict]:
    filename = path.name
    doc_ids = parse_doc_ids(filename)
    if doc_ids is None:
        raise ValueError(f"cannot parse Doc ID from filename: {filename}")
    missing = [d for d in doc_ids if d not in rule_index]
    if missing:
        raise ValueError(f"Doc ID(s) {missing} have no taxonomy entry (filename: {filename})")

    if len(doc_ids) == 1:
        doc_id = doc_ids[0]
        entry = rule_index[doc_id]
    else:
        doc_id = "&".join(f"Doc{d:02d}" for d in doc_ids)
        entry = merge_rule_entries(doc_ids, rule_index)

    rule_inputs = build_rule_inputs(entry["rules"])
    doc_meta = {
        "filename": filename,
        "doc_id": doc_id,
        "document_type": entry["document_type"],
        "standards": entry["standards"],
    }

    doc_text = extract_pdf_text(path)
    if not doc_text:
        raise ValueError(f"no extractable text in {filename} (possibly scanned; needs OCR)")

    extractions = extract_iros(doc_text, rule_inputs, doc_meta)

    return [
        {
            "filename": filename,
            "doc_id": doc_id,
            "document_type": entry["document_type"],
            "rule_id": e.rule_id,
            "evidence_status": e.evidence_status.value,
            "supporting_excerpt": e.supporting_excerpt,
            "reasoning": e.reasoning,
        }
        for e in extractions
    ]


def run_corpus(corpus_dir: str = CORPUS_DIR) -> tuple[list[dict], list[str]]:
    rule_index = load_pipeline_context()
    pdf_paths = sorted(Path(corpus_dir).glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDF file(s) in {corpus_dir}")

    all_results = []
    failed_documents = []

    for i, path in enumerate(pdf_paths, start=1):
        start = time.time()
        try:
            records = process_document(path, rule_index)
        except (ValueError, IROExtractionError) as e:
            print(f"[{i}/{len(pdf_paths)}] FAILED {path.name}: {e}")
            failed_documents.append(path.name)
            continue
        elapsed = time.time() - start
        all_results.extend(records)
        print(f"[{i}/{len(pdf_paths)}] DONE {path.name} in {elapsed:.1f}s ({len(records)} results)")

    print()
    n_ok = len(pdf_paths) - len(failed_documents)
    print(f"Processed {len(pdf_paths)} file(s): {n_ok} succeeded, {len(failed_documents)} failed.")
    if failed_documents:
        print("Failed documents (rerun manually with run_single_document()):")
        for f in failed_documents:
            print(f"  - {f}")

    return all_results, failed_documents


def run_single_document(filename: str, corpus_dir: str = CORPUS_DIR) -> list[dict]:
    rule_index = load_pipeline_context()
    path = Path(corpus_dir) / filename
    return process_document(path, rule_index)


if __name__ == "__main__":
    results, failed = run_corpus()
    print(f"\nTotal extraction records: {len(results)}")
