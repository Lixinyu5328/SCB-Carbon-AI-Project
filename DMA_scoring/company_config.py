"""
Shared company-selection helper for this folder's scripts.

Run any script here with `--company <name>` (e.g. `--company sap`,
`--company puma`) and every default input/output filename plus the
company-document corpus directory will be derived from it automatically -
no more hand-editing hardcoded paths when switching companies. Any
individual path can still be overridden explicitly via its own flag
(e.g. --corpus-dir, --input-csv, --evidence).
"""

import argparse

DEFAULT_COMPANY = "sap"
DOCUMENTS_ROOT = "/Users/lixinyu/Documents/CarbonAI_code"


def get_company() -> str:
    """Reads --company from sys.argv (falls back to DEFAULT_COMPANY).
    Uses parse_known_args so this never conflicts with a script's own
    argparse setup, regardless of parsing order."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--company", default=DEFAULT_COMPANY)
    known_args, _ = pre_parser.parse_known_args()
    return known_args.company.strip().lower()


def corpus_dir_for(company: str) -> str:
    """Company document corpus directory, e.g.
    '.../SAP_IROs_extraction/SAP_documents'. If a specific company's real
    folder does not follow this <TAG>_IROs_extraction/<TAG>_documents
    pattern, pass --corpus-dir explicitly instead."""
    tag = company.upper()
    return f"{DOCUMENTS_ROOT}/{tag}_IROs_extraction/{tag}_documents"
