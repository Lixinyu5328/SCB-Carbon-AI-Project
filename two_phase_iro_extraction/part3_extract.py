import json
import time
import os

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from part2_schemas import EvidenceStatus, IROExtractionBatch, IROExtractionResult, RuleInput

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, ValueError)


class IROExtractionError(Exception):
    pass

SYSTEM_PROMPT = """You are an ESRS (European Sustainability Reporting Standards) IRO identification expert supporting a Double Materiality Assessment. You will be given the content of a company document and a list of IRO rules. For each rule, decide whether the document contains evidence that the rule is triggered.

When deciding whether evidence supports a rule, verify two things:
(1)The evidence names the specific resource, practice, technology, group, or scope that
rule_statement requires — not merely a related or thematically adjacent one under the same
ESRS topic, standard, or sub-subtopic label. Evidence about one specific practice does not
support a rule about a different specific practice, even if both sound related.
(2)For negative_impact/risk rules, check that the evidence shows the deficient/harmful/absent 
condition described, not its opposite (adequate governance, effective controls, functioning 
processes). Also distinguish (a) a stated policy/commitment/management measure from (b) 
evidence that the actual condition has occurred — type (a) alone is insufficient unless 
rule_statement's own condition concerns the existence of a policy itself.
If the evidence fails either check for the rule as stated, set evidence_status to "not_found"
rather than forcing a mismatched tag.

For each rule, set evidence_status to one of:
- "explicit": the document clearly and directly describes the situation in rule_statement.
- "implied": the document contains evidence that reasonably suggests the rule applies, even without an exact or explicit statement. The required level of specificity depends on what rule_statement itself asks for:
  * If rule_statement's own condition concerns the existence of a policy, commitment, or action (e.g. "if company documents show policies or actions to..."), a genuine, substantive statement naming that policy or action is sufficient evidence, even without further measured outcomes.
  * If rule_statement requires something beyond policy existence -- credible performance, a measurable outcome, a specific technology or investment, or defined scope/frequency/coverage -- then a bare policy statement or generic aspirational language alone is NOT sufficient; the excerpt must name the specific practice, outcome, or measure that rule_statement requires.
  In all cases, a zero-content slogan or vague aspirational phrase with no identifiable policy, action, practice, or outcome named (e.g. "we value sustainability", "circularity is a focus area") is not sufficient on its own.
- "not_found": the document contains no relevant evidence for this rule, or the only relevant text is generic/aspirational language lacking the specificity described above.

The identification_cues field lists example terms and phrases, but do not treat them as required keywords. A rule should be matched if the document contains semantically equivalent evidence, even if none of the listed cues appear verbatim. Use rule_statement as the primary matching criterion, and treat identification_cues as illustrative examples of what relevant evidence might look like.

For every rule, whether matched or not, return a supporting_excerpt: if evidence_status is "explicit" or "implied", quote or closely paraphrase the specific passage that supports your decision; if evidence_status is "not_found", return an empty string. Always include a one-sentence reasoning explaining your decision.

You must return exactly one result per rule_id provided, using the same rule_id values. Do not skip any rule and do not invent rule_ids that were not provided."""

def build_user_prompt(doc_text: str, rules: list[RuleInput], doc_meta: dict) -> str:
    rules_json = json.dumps([r.model_dump() for r in rules], ensure_ascii=False)
    return (
        f"Document type: {doc_meta.get('document_type', 'unknown')}\n"
        f"Document filename: {doc_meta.get('filename', 'unknown')}\n\n"
        f'Document content:\n"""\n{doc_text}\n"""\n\n'
        f"Rules to evaluate ({len(rules)} total):\n{rules_json}"
    )


def reconcile_extractions(extractions: list[IROExtractionResult], rules: list[RuleInput]) -> list[IROExtractionResult]:
    expected_ids = {r.rule_id for r in rules}
    returned_ids = {e.rule_id for e in extractions}

    reconciled = [e for e in extractions if e.rule_id in expected_ids]

    missing_ids = expected_ids - returned_ids
    for rule_id in missing_ids:
        reconciled.append(
            IROExtractionResult(
                rule_id=rule_id,
                evidence_status=EvidenceStatus.not_found,
                supporting_excerpt="",
                reasoning="No result returned by model for this rule; defaulted to not_found.",
            )
        )

    dropped_ids = returned_ids - expected_ids
    if dropped_ids:
        print(f"[reconcile_extractions] dropped {len(dropped_ids)} hallucinated rule_id(s): {sorted(dropped_ids)}")
    if missing_ids:
        print(f"[reconcile_extractions] {len(missing_ids)} rule_id(s) missing from model output, defaulted to not_found: {sorted(missing_ids)}")

    return reconciled


def extract_iros(doc_text: str, rules: list[RuleInput], doc_meta: dict) -> list[IROExtractionResult]:
    user_prompt = build_user_prompt(doc_text, rules, doc_meta)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=IROExtractionBatch,
            )
            if response.output_parsed is None:
                raise ValueError("model returned no parsed output (possible refusal)")
            return reconcile_extractions(response.output_parsed.extractions, rules)
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"[extract_iros] {doc_meta.get('filename', '?')} attempt {attempt} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)

    print(f"[extract_iros] {doc_meta.get('filename', '?')} failed after {MAX_RETRIES} attempts.")
    raise IROExtractionError(f"extract_iros failed for {doc_meta.get('filename', '?')} after {MAX_RETRIES} attempts: {last_error}")
