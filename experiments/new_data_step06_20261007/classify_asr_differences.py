"""Read-only listening triage of saved ASR differences; no label corrections."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
from pathlib import Path
import re

import asr_audit as audit


def number_forms(digits):
    """Conservative integer spelling candidates, solely for human-review triage."""
    if not re.fullmatch(r"[0-9]{1,4}", digits):
        return set()
    number = int(digits)
    sino = "영일이삼사오육칠팔구"
    forms = {"".join(sino[int(digit)] for digit in digits),
             "".join("공" if digit == "0" else sino[int(digit)] for digit in digits)}
    if number == 0:
        forms.update(("영", "공", "제로"))
    if not (len(digits) > 1 and digits.startswith("0")):
        text = ""
        for unit, label in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
            amount = number // unit % 10
            if amount:
                text += ("" if amount == 1 and unit > 1 else sino[amount]) + label
        if text:
            forms.add(text)
        if 1 <= number <= 99:
            tens = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
            ones = ("", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉")
            count = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
            forms.update((tens[number // 10] + ones[number % 10], tens[number // 10] + count[number % 10]))
            if number == 20:
                forms.add("스무")
    return forms


def numeric_pair(first, second):
    return (first.isascii() and first.isdigit() and second in number_forms(first)) or (
        second.isascii() and second.isdigit() and first in number_forms(second))


def classify(record):
    ref, prediction = record["normalized_reference"], record["normalized_asr"]
    if record["status"] != "ok":
        return "asr_unavailable_requires_listening", []
    if ref == prediction:
        return "normalized_match", []
    left, right = ref.replace(" ", ""), prediction.replace(" ", "")
    if left == right:
        return "spacing_only_difference", []
    differences = [{"operation": kind, "reference_part": left[a:b], "asr_part": right[c:d],
                    "reference_span_without_spaces": [a, b], "asr_span_without_spaces": [c, d]}
                   for kind, a, b, c, d in difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
                   if kind != "equal"]
    has_decimal_or_grouping = re.search(r"[0-9][.,][0-9]", record["reference"] + " " + record["asr_text"])
    if not has_decimal_or_grouping and differences and all(numeric_pair(item["reference_part"], item["asr_part"]) for item in differences):
        return "possible_number_expression_difference", differences
    return "possible_content_difference", differences


def build(source, source_ref):
    audit.require(source["schema"] == "step06-local-asr-audit-results-v1" and source["status"] == "completed"
                  and len(source["records"]) == 120 and source["ids"] == [row["id"] for row in source["records"]]
                  and len(set(source["ids"])) == 120, "Need complete ordered120 ASR records")
    records = []
    for row in source["records"]:
        comparison = audit.compare_text(row["reference"], row["asr_text"])
        audit.require(all(row[key] == value for key, value in comparison.items()), "Saved ASR comparison differs")
        category, differences = classify(row)
        priority = (1 if row["boundary_flags"] or category in ("possible_content_difference", "asr_unavailable_requires_listening")
                    else 2 if category == "possible_number_expression_difference" else 3 if category == "spacing_only_difference" else 4)
        records.append({"id": row["id"], "split": row["split"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
            "reference": row["reference"], "asr_text": row["asr_text"], "normalized_reference": row["normalized_reference"],
            "normalized_asr": row["normalized_asr"], "audio": row["audio"], "asr_status": row["status"],
            "category": category, "suggested_listening_priority": priority, "difference_spans": differences,
            "boundary_flags": row["boundary_flags"], "confidence_flags": row["confidence_flags"],
            "original_asr_review_status": row["review_status"], "metrics": row["metrics"],
            "human_listening_confirmed": False, "training_ready": False, "label_modified": False})
    return {"schema": "step06-asr-listening-triage-v1", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "asr_results": source_ref, "ids": source["ids"], "record_count": 120, "records": records,
        "mismatches_in_original_order": [row for row in records if row["category"] != "normalized_match"],
        "category_counts": dict(Counter(row["category"] for row in records)),
        "policy": "Keep original120 ordering; assign review hints without changing any ASR gate, transcript, split or training readiness",
        "integer_heuristic": "Only 1-4 ASCII digit integers compared to conservative Korean Sino/native forms; every changed span must match; decimals/grouping are excluded",
        "priority_policy": "1 boundary/content/unavailable; 2 possible integer-expression difference; 3 spacing-only; 4 normalized match; original order is never rearranged",
        "human_listening_confirmed_count": 0, "labels_modified": False, "model_executed": False,
        "limits": ["Possible number-expression differences are hypotheses for listening, not proof that the words or quantities match",
                   "Content-difference category can still reflect ASR errors rather than wrong source labels",
                   "Existing ASR decisions and original transcripts remain authoritative evidence, unchanged by this triage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=audit.RESULT)
    parser.add_argument("--output", type=Path, default=audit.BASE / "results/asr_difference_review.json")
    args = parser.parse_args()
    ref = audit.bind(args.input)
    value = build(audit.read(args.input), ref)
    audit.verify(ref)
    value["code"] = audit.bind(Path(__file__))
    value["tests"] = audit.bind(Path(__file__).with_name("test_classify_asr_differences.py"))
    audit.save_new(args.output, value)
    print(value["category_counts"])


if __name__ == "__main__":
    main()
