from enum import Enum

from pydantic import BaseModel


class IROType(str, Enum):
    negative_impact = "negative_impact"
    positive_impact = "positive_impact"
    risk = "risk"
    opportunity = "opportunity"


class EvidenceStatus(str, Enum):
    explicit = "explicit"
    implied = "implied"
    not_found = "not_found"


class RuleInput(BaseModel):
    rule_id: str
    esrs_topic: str
    esrs_subtopic: str
    esrs_sub_subtopic: str
    iro_type: IROType
    rule_statement: str
    identification_cues: str


class IROExtractionResult(BaseModel):
    rule_id: str
    evidence_status: EvidenceStatus
    supporting_excerpt: str
    reasoning: str


class IROExtractionBatch(BaseModel):
    extractions: list[IROExtractionResult]


def to_rule_input(rule: dict) -> RuleInput:
    return RuleInput(
        rule_id=rule["rule_id"],
        esrs_topic=rule["esrs_topic"],
        esrs_subtopic=rule["esrs_subtopic"],
        esrs_sub_subtopic=rule["esrs_sub_subtopic"],
        iro_type=rule["iro_type"],
        rule_statement=rule["rule_statement"],
        identification_cues=rule["identification_cues"],
    )


def build_rule_inputs(rules: list[dict]) -> list[RuleInput]:
    return [to_rule_input(r) for r in rules]


if __name__ == "__main__":
    import json

    from part1_taxonomy_mapping import build_rule_index, load_rulebook, load_taxonomy

    standard_rules = load_rulebook("IRO_rulebook_Final_1.json")
    taxonomy = load_taxonomy("Document_taxonomy_1.docx")
    rule_index = build_rule_index(standard_rules, taxonomy)

    for doc_id in (1, 3):
        entry = rule_index[doc_id]
        rule_inputs = build_rule_inputs(entry["rules"])
        payload = [r.model_dump() for r in rule_inputs]
        char_len = len(json.dumps(payload, ensure_ascii=False))
        est_tokens = char_len // 4
        print(f"Doc{doc_id:02d} | {entry['document_type']:<40} | rules={len(rule_inputs)} | ~{est_tokens} tokens (est.)")

    print()
    print("IROExtractionBatch JSON schema:")
    print(json.dumps(IROExtractionBatch.model_json_schema(), indent=2))
