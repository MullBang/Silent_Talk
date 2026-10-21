"""Training-observed token/syllable/eojeol coverage, using stored label metadata."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/offline_unseen_phrase_step05d_20261018"
TRAIN = ROOT / "experiments/offline_phrase_only_step05c_20261018/runs/seed42/evaluations/update_1680/evaluation_summary.json"
DEV = BASE / "runs/seed42/evaluation_summary.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


ev = module("stage5d_coverage_helpers", BASE / "evaluation/evaluator.py")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def symbols(text):
    nfc = ev.contract.metric_text(text)
    return [c for c in nfc if "\uac00" <= c <= "\ud7a3"]


def coverage(items, seen):
    known = [item for item in items if item in seen]
    unseen = Counter(item for item in items if item not in seen)
    return {"occurrences": len(items), "seen_occurrences": len(known), "occurrence_coverage": len(known) / len(items) if items else None,
            "unique": len(set(items)), "seen_unique": len(set(items) & set(seen)),
            "unique_coverage": len(set(items) & set(seen)) / len(set(items)) if items else None,
            "unseen": [{"unit": unit, "occurrences": count} for unit, count in sorted(unseen.items())]}


def main():
    train = [r for r in read(TRAIN)["rows"] if r["condition"] == "normal"]
    dev = [r for r in read(DEV)["rows"] if r["condition"] == "normal"]
    assert len(train) == 12 and len(dev) == 3
    assert all(r["split"] == "train" and r["kind"] == "phrase" for r in train)
    assert all(r["split"] == "dev" and r["kind"] == "phrase" for r in dev)
    train_tokens = [t for r in train for t in r["reference_label_ids"]]
    train_syllables = [c for r in train for c in symbols(r["reference_text"])]
    train_eojeols = [word for r in train for word in ev.contract.metric_text(r["reference_text"]).split()]
    vocabulary = ev.contract.IDX_TO_GRAPHEME
    assert len(vocabulary) == 69 and 0 in vocabulary
    for r in train + dev:
        decoded = ev.contract.compose_jamos((vocabulary[t] for t in r["reference_label_ids"]), strict=True)
        assert ev.contract.metric_text(decoded) == ev.contract.metric_text(r["reference_text"])
    out = []
    for r in dev:
        tokens = r["reference_label_ids"]
        speaker = r["parent_id"].split("_")[5]
        speaker_train = [t for t in train if t["parent_id"].split("_")[5] == speaker]
        token_coverage = coverage(tokens, set(train_tokens))
        for item in token_coverage["unseen"]:
            item["grapheme"] = vocabulary[item["unit"]]
        out.append({"clip_id": r["clip_id"], "speaker_id": speaker, "reference_text": r["reference_text"],
                    "vocabulary_representable": all(t in vocabulary and t != 0 for t in tokens),
                    "tokens_including_space": token_coverage,
                    "hangul_syllables_excluding_spaces": coverage(symbols(r["reference_text"]), set(train_syllables)),
                    "eojeols_exact_space_delimited": coverage(ev.contract.metric_text(r["reference_text"]).split(), set(train_eojeols)),
                    "same_speaker_train_clip_count": len(speaker_train),
                    "same_speaker_observed_token_coverage": coverage(tokens, {v for t in speaker_train for v in t["reference_label_ids"]}),
                    "same_speaker_observed_eojeol_coverage": coverage(ev.contract.metric_text(r["reference_text"]).split(),
                                                                     {word for t in speaker_train for word in ev.contract.metric_text(t["reference_text"]).split()})})
    all_tokens = [v for r in dev for v in r["reference_label_ids"]]
    all_syllables = [c for r in dev for c in symbols(r["reference_text"])]
    all_eojeols = [w for r in dev for w in ev.contract.metric_text(r["reference_text"]).split()]
    overall = {"vocabulary_representable_all": all(r["vocabulary_representable"] for r in out),
               "tokens_including_space": coverage(all_tokens, set(train_tokens)),
               "hangul_syllables_excluding_spaces": coverage(all_syllables, set(train_syllables)),
               "eojeols_exact_space_delimited": coverage(all_eojeols, set(train_eojeols))}
    for item in overall["tokens_including_space"]["unseen"]:
        item["grapheme"] = vocabulary[item["unit"]]
    phrase_speakers = defaultdict(set)
    speaker_phrases = defaultdict(set)
    for r in train:
        text = ev.contract.metric_text(r["reference_text"])
        speaker = r["parent_id"].split("_")[5]
        phrase_speakers[text].add(speaker)
        speaker_phrases[speaker].add(text)
    result = {"schema": "offline-stage5d-observed-label-coverage-v1", "client_date": "2026-10-18", "status": "completed_metadata_only",
              "sources": [ev.ref(TRAIN), ev.ref(DEV)], "analysis_code": ev.ref(Path(__file__)),
              "vocabulary": {"total_size_including_blank": 69, "reference_blank_allowed": False,
                             "nonblank_symbol_inventory": 68, "observed_train_unique_tokens": len(set(train_tokens)),
                             "observed_train_token_occurrences": len(train_tokens)},
              "train_counts": {"clips": 12, "unique_phrase_texts": len(phrase_speakers), "speakers": len(speaker_phrases),
                               "hangul_syllables": len(train_syllables), "unique_hangul_syllables": len(set(train_syllables)),
                               "eojeols": len(train_eojeols), "unique_eojeols": len(set(train_eojeols))},
              "overall_dev3": overall, "per_dev_clip": out,
              "train_phrase_speaker_distribution": [{"text": text, "speaker_count": len(speakers), "speakers": sorted(speakers)}
                                                    for text, speakers in sorted(phrase_speakers.items())],
              "train_speaker_phrase_distribution": [{"speaker_id": speaker, "unique_phrase_count": len(texts), "texts": sorted(texts)}
                                                    for speaker, texts in sorted(speaker_phrases.items())],
              "definitions": {
                  "tokens": "Actual stored69-vocabulary reference IDs, including inter-word space; exclude blank0 from references.",
                  "hangul_syllable": "Each NFC precomposed Hangul syllable U+AC00..U+D7A3; spaces are excluded.",
                  "eojeol": "Exact whitespace-delimited NFC word form; no morphology, pronunciation or semantic matching.",
                  "observed_coverage": "Membership in the12 training-phrase labels; no estimate of visual recognition or adequate exposure.",
                  "same_speaker": "Post-hoc grouping with ground-truth speaker metadata, unavailable as an inferred recognition output.",
                  "independence": "The three dev label sequences are counted once, not twice for seeds or three times for controls."},
              "limits_ko": [
                  "69토큰으로표현가능하다는사실과학습12구절에서해당토큰을관측했다는사실은다르다. 관측했더라도새조합의음절·어절을읽을성능은보장하지않는다.",
                  "음절·어절미관측비율은작은자료의범위를설명할뿐이번실패의유일원인을증명하지않는다. 전처리·시각특징·학습·디코딩의기여를이통계로분리할수없다.",
                  "같은화자의여러구절및같은구절의여러화자확보를위한자료구성조사를지원하는메타데이터통계이며추가학습·자료승격의근거는아니다."],
              "new_neural_calls": 0, "new_optimizer_updates": 0, "pixel_arrays_read": 0, "checkpoint_tensor_loads": 0,
              "new_downloads": 0, "service_changes": 0}
    ev.write_json(BASE / "analysis/coverage_analysis.json", result)
    print(json.dumps({"vocabulary": result["vocabulary"], "train": result["train_counts"], "dev": overall}, ensure_ascii=False))


if __name__ == "__main__": main()
