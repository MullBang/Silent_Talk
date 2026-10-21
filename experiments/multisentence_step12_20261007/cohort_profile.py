"""Post-selection metadata profile; no pixels, features, models or outcomes read.

This describes the already frozen32 clips. It cannot choose or change clips,
training settings, stopping rules, checkpoints, or evaluation results.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import unicodedata

BASE = Path(__file__).resolve().parent
POLICY = BASE / 'diagnostic_plan.json'
FREEZE = BASE / 'plans/freeze_manifest.json'
OUTPUT = BASE / 'results/cohort_profile.json'
EXPECTED_POLICY_SHA = '2fc10937d9c9a879e62839cac57bbe9337319dff80a3bdf71b5f344f1d701063'
EXPECTED_FREEZE_SHA = '21dba58f2d15c12141d1428c7ea8914db1c29d8fd2d7c63623f21223c0e94b26'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read_bound(path, expected_sha):
    require(sha(path) == expected_sha, 'Frozen source changed: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    require(values, 'No values to describe')
    return {'n': len(values), 'total': sum(values), 'minimum': min(values), 'maximum': max(values),
            'mean': statistics.fmean(values), 'median': statistics.median(values),
            'population_standard_deviation': statistics.pstdev(values)}


def units(row, kind):
    text = unicodedata.normalize('NFC', row['text'])
    if kind == 'token_ids':
        return list(row['tokens'])
    if kind == 'hangul_syllables':
        return [character for character in text if '\uac00' <= character <= '\ud7a3']
    require(kind == 'whitespace_delimited_words', 'Unknown unit kind')
    return text.split()


def vocabulary(rows, kind):
    return set(value for row in rows for value in units(row, kind))


def novelty(row, training_rows, kind):
    values = units(row, kind)
    training_vocabulary = vocabulary(training_rows, kind)
    missing = sorted(set(values) - training_vocabulary)
    missing_set = set(missing)
    occurrence_count = sum(value in missing_set for value in values)
    return {'total_occurrences': len(values), 'unique_values': len(set(values)),
            'training_unique_values': len(training_vocabulary), 'unseen_unique_values': missing,
            'unseen_unique_count': len(missing), 'unseen_occurrences': occurrence_count,
            'unseen_occurrence_fraction': occurrence_count / len(values) if values else None}


def group_profile(rows):
    tokens = [len(row['tokens']) for row in rows]
    frames = [row['frames'] for row in rows]
    return {'utterances': len(rows), 'speakers': len({row['speaker_id'] for row in rows}),
            'recordings': len({row['source_stem'] for row in rows}), 'frames': stats(frames),
            'frame_count_divided_by_25_seconds': stats([value / 25 for value in frames]),
            'target_tokens': stats(tokens), 'ctc_minimum_frames': stats([row['ctc_min_input_frames'] for row in rows]),
            'ctc_slack_frames': stats([row['ctc_slack'] for row in rows]),
            'target_tokens_per_frame': stats([len(row['tokens']) / row['frames'] for row in rows]),
            'hangul_syllable_occurrences': stats([len(units(row, 'hangul_syllables')) for row in rows]),
            'whitespace_delimited_word_occurrences': stats([len(units(row, 'whitespace_delimited_words')) for row in rows]),
            'token_class_ids_present': sorted(vocabulary(rows, 'token_ids')),
            'token_class_histogram': {str(key): count for key, count in sorted(Counter(token for row in rows for token in row['tokens']).items())},
            'hangul_syllable_types': len(vocabulary(rows, 'hangul_syllables')),
            'whitespace_delimited_word_types': len(vocabulary(rows, 'whitespace_delimited_words'))}


def aggregate_novelty(records, comparison, kind):
    rows = [row['coverage'][comparison][kind] for row in records]
    total = sum(row['total_occurrences'] for row in rows)
    missing = sum(row['unseen_occurrences'] for row in rows)
    return {'utterances': len(rows), 'utterances_with_unseen_values': sum(row['unseen_unique_count'] > 0 for row in rows),
            'total_occurrences': total, 'unseen_occurrences': missing,
            'unseen_occurrence_fraction': missing / total if total else None,
            'union_of_unseen_values': sorted(set(value for row in rows for value in row['unseen_unique_values'])),
            'sum_per_utterance_unseen_unique_counts': sum(row['unseen_unique_count'] for row in rows)}


def build_profile(policy):
    require(policy['schema'] == 'lipnet-step12-multisentence-protocol-v1', 'Unexpected frozen policy')
    selection = policy['selection']
    rows = {row['id']: row for row in selection['records']}
    train_ids, hold_ids = selection['train_ids'], selection['holdout_ids']
    require(len(rows) == len(selection['records']) == 32 and len(train_ids) == 24 and len(hold_ids) == 8 and
            set(train_ids).isdisjoint(hold_ids) and set(rows) == set(train_ids + hold_ids), 'Wrong frozen cohort')
    train, hold = [rows[sid] for sid in train_ids], [rows[sid] for sid in hold_ids]
    normalized = [''.join(unicodedata.normalize('NFC', row['text']).split()) for row in rows.values()]
    require(len(set(normalized)) == 32, 'Normalized target collision')
    for row in rows.values():
        require(row['original_split'] == 'train' and row['diagnostic_role'] ==
                ('train' if row['id'] in train_ids else 'internal_holdout'), 'Changed source/diagnostic role')
        require(row['tokens'] and all(type(token) is int and 1 <= token < 69 for token in row['tokens']),
                'Invalid nonblank target IDs')
        minimum = len(row['tokens']) + sum(a == b for a, b in zip(row['tokens'], row['tokens'][1:]))
        require(minimum == row['ctc_min_input_frames'] and row['ctc_slack'] == row['frames'] - minimum >= 0,
                'CTC minimum/slack mismatch')
    kinds = ('token_ids', 'hangul_syllables', 'whitespace_delimited_words')
    holdout_records = []
    for row in hold:
        own = [candidate for candidate in train if candidate['speaker_id'] == row['speaker_id']]
        require(len(own) == 3, 'Expected three training utterances per internal holdout speaker')
        holdout_records.append({'id': row['id'], 'speaker_id': row['speaker_id'], 'generation': row['generation'],
                                'source_stem': row['source_stem'], 'text': row['text'], 'frames': row['frames'],
                                'frame_count_divided_by_25_seconds': row['frames'] / 25,
                                'target_tokens': len(row['tokens']), 'ctc_minimum_frames': row['ctc_min_input_frames'],
                                'ctc_slack_frames': row['ctc_slack'], 'own_speaker_training_ids': [candidate['id'] for candidate in own],
                                'coverage': {comparison: {kind: novelty(row, training_rows, kind) for kind in kinds}
                                             for comparison, training_rows in (('all_train24', train), ('own_speaker_train3', own))}})
    train_classes, hold_classes = vocabulary(train, 'token_ids'), vocabulary(hold, 'token_ids')
    groups = selection['evaluation_groups']
    return {'groups': {name: group_profile([rows[sid] for sid in ids]) for name, ids in groups.items()},
            'target_class_coverage': {'train_class_ids': sorted(train_classes), 'holdout_class_ids': sorted(hold_classes),
                                      'holdout_class_ids_absent_from_all_train24': sorted(hold_classes - train_classes),
                                      'all_holdout_target_classes_present_in_train24': hold_classes <= train_classes},
            'holdout_records': holdout_records,
            'holdout_novelty': {comparison: {kind: aggregate_novelty(holdout_records, comparison, kind) for kind in kinds}
                                for comparison in ('all_train24', 'own_speaker_train3')},
            'normalization_duplicate_count': len(normalized) - len(set(normalized)),
            'ctc_infeasible_utterances': 0,
            'nonspace_characters_outside_hangul_syllables': sorted(set(character for row in rows.values()
                for character in unicodedata.normalize('NFC', row['text']) if not character.isspace() and not '\uac00' <= character <= '\ud7a3')),
            'definitions': {'duration': 'Frame count divided by25fps; frame-equivalent duration, not remeasured source annotation duration',
                            'ctc_slack': 'T minus target-token count minus adjacent identical-target-token pair count',
                            'token_classes': 'Raw nonblank target IDs from frozen labels; no token-class renaming or tokenizer import',
                            'syllables': 'NFC precomposed Hangul syllables U+AC00 through U+D7A3',
                            'words': 'NFC text split by Unicode whitespace; surface forms, not morphological or dictionary words',
                            'novel_occurrences': 'Count every occurrence whose exact value is absent from the specified training vocabulary',
                            'own_speaker_union': 'Union of values missing relative to each holdout speaker own three train utterances; not a global vocabulary claim'},
            'interpretation_limits': [
                'This is a post-selection descriptive profile, generated after the scope was frozen and never used to choose clips or tune models.',
                'Unseen words or Hangul syllables are not mechanically impossible for jamo-level CTC; it can compose new strings from target classes.',
                'A target class absent from positive training labels lacks positive target supervision here, but that is not a proof of zero output probability.',
                'Internal holdout speakers appear in training; recordings and normalized target sentences differ. Historical models have used these original train-pool clips.',
                'Metadata feasibility and vocabulary coverage do not establish audiovisual alignment, label correctness, recognition accuracy, or generalization.',
                'No model output, checkpoint, posterior, pixel payload, teacher feature, raw media, or human approval was read or changed.']}


def main():
    source_ref = bind(__file__)
    policy = read_bound(POLICY, EXPECTED_POLICY_SHA)
    freeze = read_bound(FREEZE, EXPECTED_FREEZE_SHA)
    require(freeze['experiment_plan'] == bind(POLICY) and freeze['status'] == 'frozen', 'Unbound frozen policy')
    result = {'schema': 'lipnet-step12-frozen-cohort-profile-v1', 'status': 'described_without_reselection',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': source_ref,
              'source_policy': bind(POLICY), 'source_freeze_manifest': bind(FREEZE),
              'post_selection_descriptive': True, 'used_for_selection_or_tuning': False,
              'source_payloads_opened': 0, 'model_outputs_read': 0, 'optimization_steps': 0,
              **build_profile(policy)}
    require(bind(__file__) == source_ref and sha(POLICY) == EXPECTED_POLICY_SHA and sha(FREEZE) == EXPECTED_FREEZE_SHA,
            'Source changed while profiling')
    with OUTPUT.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': bind(OUTPUT), 'generator': source_ref,
                      'target_class_coverage': result['target_class_coverage'],
                      'holdout_novelty': result['holdout_novelty']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
