import json
import re
from pathlib import Path

from docx import Document

RULEBOOK_PATH = "Pipeline/IRO_rulebook_Final_1.json"
TAXONOMY_PATH = "Pipeline/Document_taxonomy_1.docx"

DOC_ID_PATTERN = re.compile(r"^doc0*(\d{1,2})[_\-\s]", re.IGNORECASE)

# Supports filenames spanning multiple taxonomy doc types, e.g.
# "Doc04&Doc08_...", "Doc04&Doc08&Doc11_..." (any number of "&Doc.." repeats).
MULTI_DOC_ID_PATTERN = re.compile(r"^((?:doc0*\d{1,2}&)*doc0*\d{1,2})[_\-\s]", re.IGNORECASE)
SINGLE_ID_TOKEN_PATTERN = re.compile(r"doc0*(\d{1,2})", re.IGNORECASE)


def load_rulebook(path: str) -> dict[str, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key.replace("_rules", ""): rules for key, rules in raw.items()}


def load_taxonomy(path: str) -> dict[int, dict]:
    doc = Document(path)
    table = doc.tables[0]
    taxonomy = {}
    category = None

    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        doc_num, doc_type, description, standards_raw = cells

        if not doc_num.isdigit():
            category = doc_type
            continue

        standards = []
        conditional_standards = []
        for token in standards_raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.endswith("†"):
                token = token[:-1].strip()
                conditional_standards.append(token)
            standards.append(token)

        taxonomy[int(doc_num)] = {
            "doc_id": int(doc_num),
            "document_type": doc_type,
            "category": category,
            "description": description,
            "standards": standards,
            "conditional_standards": conditional_standards,
        }

    return taxonomy


def build_rule_index(standard_rules: dict[str, list[dict]], taxonomy: dict[int, dict]) -> dict[int, dict]:
    rule_index = {}
    for doc_id, meta in taxonomy.items():
        rules = []
        for standard in meta["standards"]:
            rules.extend(standard_rules.get(standard, []))
        rule_index[doc_id] = {**meta, "rules": rules, "rule_count": len(rules)}
    return rule_index


def parse_doc_id(filename: str) -> int | None:
    stem = Path(filename).stem
    match = DOC_ID_PATTERN.match(stem + "_")
    if not match:
        return None
    doc_id = int(match.group(1))
    return doc_id if 1 <= doc_id <= 15 else None


def parse_doc_ids(filename: str) -> list[int] | None:
    """Parse one or more Doc IDs from a filename. Supports both the original
    single-ID convention (e.g. "Doc07_...") and the "&"-joined multi-type
    convention used for documents spanning more than one taxonomy entry
    (e.g. "Doc04&Doc08_..." or "Doc04&Doc08&Doc11_...", any number of IDs).
    Returns the IDs in the order they appear, or None if the filename cannot
    be parsed, contains an out-of-range ID, or repeats the same ID twice."""
    stem = Path(filename).stem
    match = MULTI_DOC_ID_PATTERN.match(stem + "_")
    if not match:
        return None

    doc_ids = [int(m.group(1)) for m in SINGLE_ID_TOKEN_PATTERN.finditer(match.group(1))]
    if not doc_ids:
        return None
    if any(not (1 <= d <= 15) for d in doc_ids):
        return None
    if len(doc_ids) != len(set(doc_ids)):
        return None

    return doc_ids


def merge_rule_entries(doc_ids: list[int], rule_index: dict[int, dict]) -> dict:
    """Combine the taxonomy entries for two or more doc_ids into a single
    entry with the same shape as a normal rule_index value (document_type,
    category, description, standards, conditional_standards, rules,
    rule_count), so a multi-type document can be processed with one merged
    rule set instead of being forced into a single type. Rules are
    deduplicated by rule_id, since two doc types that share an ESRS standard
    would otherwise include the exact same rules twice."""
    entries = [rule_index[d] for d in doc_ids]

    merged_rules_by_id: dict[str, dict] = {}
    for entry in entries:
        for rule in entry["rules"]:
            merged_rules_by_id.setdefault(rule["rule_id"], rule)
    merged_rules = list(merged_rules_by_id.values())

    return {
        "doc_id": doc_ids,
        "document_type": " + ".join(entry["document_type"] for entry in entries),
        "category": " / ".join(dict.fromkeys(entry["category"] for entry in entries)),
        "description": " | ".join(entry["description"] for entry in entries),
        "standards": list(dict.fromkeys(s for entry in entries for s in entry["standards"])),
        "conditional_standards": list(
            dict.fromkeys(s for entry in entries for s in entry["conditional_standards"])
        ),
        "rules": merged_rules,
        "rule_count": len(merged_rules),
    }


def get_rules_for_document(filename: str, rule_index: dict[int, dict]) -> dict:
    doc_ids = parse_doc_ids(filename)
    if doc_ids is None:
        raise ValueError(f"Cannot parse Doc ID from filename: {filename}")
    missing = [d for d in doc_ids if d not in rule_index]
    if missing:
        raise KeyError(f"Doc ID(s) {missing} have no taxonomy entry")
    if len(doc_ids) == 1:
        return rule_index[doc_ids[0]]
    return merge_rule_entries(doc_ids, rule_index)


if __name__ == "__main__":
    standard_rules = load_rulebook(RULEBOOK_PATH)
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    rule_index = build_rule_index(standard_rules, taxonomy)

    print(f"Standards loaded: {list(standard_rules.keys())}")
    print(f"Total rules: {sum(len(v) for v in standard_rules.values())}")
    print(f"Doc types loaded: {len(taxonomy)}")
    print()

    for doc_id in sorted(rule_index):
        entry = rule_index[doc_id]
        flag = " (conditional: " + ", ".join(entry["conditional_standards"]) + ")" if entry["conditional_standards"] else ""
        print(f"Doc{doc_id:02d} | {entry['document_type']:<55} | {entry['standards']}{flag} | rules={entry['rule_count']}")

    print()
    test_names = ["Doc01_real_climate-policy.txt", "Doc1_synth_gapfill.txt", "Doc07_real_hs-procedures.pdf", "Doc15.txt"]
    for name in test_names:
        doc_id = parse_doc_id(name)
        print(f"{name} -> doc_id={doc_id}")
