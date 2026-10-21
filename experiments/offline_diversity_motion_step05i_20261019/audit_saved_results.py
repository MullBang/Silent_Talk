"""Independent saved-result audit: NumPy only, no model/checkpoint imports.

Reconstruct input transforms, numeric-ID Hangul decoding and full-matrix edit
distances independently of the training runner and evaluation contract.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
import unicodedata
from typing import Any
import numpy as np
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]

def read(path: str | Path) -> dict[str, Any]:
    """Read a saved JSON record; IO/format failures surface to the audit boundary."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as error:
        error.add_note('Independent saved-result audit function: read')
        raise

def check(value: Any, label: str) -> None:
    """Reject an invariant with a concrete failure label."""
    try:
        if not value:
            raise ValueError(label)
    except Exception as error:
        error.add_note('Independent saved-result audit function: check')
        raise

def digest(path: str | Path) -> str:
    """Hash saved bytes without deserializing model tensors."""
    try:
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    except Exception as error:
        error.add_note('Independent saved-result audit function: digest')
        raise

def bound(binding: dict[str, Any]) -> Path:
    """Validate file identity and return the bound path."""
    try:
        path = Path(binding['path'])
        check(digest(path) == binding['sha256'], 'File SHA mismatch: ' + str(path))
        if 'bytes' in binding:
            check(path.stat().st_size == binding['bytes'], 'File size mismatch')
        return path
    except Exception as error:
        error.add_note('Independent saved-result audit function: bound')
        raise

def array_digest(array: np.ndarray) -> str:
    """Hash contiguous array payload independently of the runner."""
    try:
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    except Exception as error:
        error.add_note('Independent saved-result audit function: array_digest')
        raise

def text(value: str) -> str:
    """Apply the declared NFC and outer-whitespace metric normalization."""
    try:
        return unicodedata.normalize('NFC', value).strip()
    except Exception as error:
        error.add_note('Independent saved-result audit function: text')
        raise

def decode(raw: list[int]) -> tuple[list[int], str]:
    """Collapse CTC and compose numeric Korean jamo while retaining malformed symbols."""
    try:
        collapsed = [int(v) for i, v in enumerate(raw) if v != 0 and (i == 0 or v != raw[i - 1])]
        output, i = ([], 0)
        while i < len(collapsed):
            first = collapsed[i]
            if first == 68:
                output.append(' ')
                i += 1
            elif 1 <= first <= 19 and i + 1 < len(collapsed) and (20 <= collapsed[i + 1] <= 40):
                vowel = collapsed[i + 1]
                i += 2
                tail = 0
                if i < len(collapsed) and 41 <= collapsed[i] <= 67:
                    tail = collapsed[i] - 40
                    i += 1
                output.append(chr(44032 + (first - 1) * 588 + (vowel - 20) * 28 + tail))
            else:
                output.append('�')
                i += 1
        return (collapsed, ''.join(output))
    except Exception as error:
        error.add_note('Independent saved-result audit function: decode')
        raise

def distance(reference: str | list[str], prediction: str | list[str]) -> int:
    """Compute full-matrix Levenshtein independently of the production scorer."""
    try:
        matrix = np.zeros((len(reference) + 1, len(prediction) + 1), np.int64)
        matrix[:, 0] = np.arange(len(reference) + 1)
        matrix[0, :] = np.arange(len(prediction) + 1)
        for i in range(1, len(reference) + 1):
            for j in range(1, len(prediction) + 1):
                matrix[i, j] = min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + int(reference[i - 1] != prediction[j - 1]))
        return int(matrix[-1, -1])
    except Exception as error:
        error.add_note('Independent saved-result audit function: distance')
        raise

def score(reference: str, prediction: str) -> dict[str, Any]:
    """Recompute every saved per-example metric."""
    try:
        ref, pred = (text(reference), text(prediction))
        maximum, run, previous = (0, 0, None)
        for char in pred:
            run = run + 1 if char == previous else 1
            previous = char
            if not char.isspace():
                maximum = max(maximum, run)
        return {'reference': ref, 'prediction': pred, 'failed': False, 'character_edits': distance(ref, pred), 'reference_characters': len(ref), 'word_edits': distance(ref.split(), pred.split()), 'reference_words': len(ref.split()), 'exact': ref == pred, 'replacement_characters': pred.count('�'), 'maximum_same_nonspace_character_run': maximum, 'repetition_ge5': maximum >= 5}
    except Exception as error:
        error.add_note('Independent saved-result audit function: score')
        raise

def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate corpus edit counts, preserving the declared denominators."""
    try:
        total = {key: sum((s[key] for s in scores)) for key in ['character_edits', 'reference_characters', 'word_edits', 'reference_words']}
        return {'count': len(scores), **total, 'CER': total['character_edits'] / total['reference_characters'], 'WER': total['word_edits'] / total['reference_words'], 'exact_count': sum((s['exact'] for s in scores)), 'failure_count': 0, 'repetition_ge5_count': sum((s['repetition_ge5'] for s in scores))}
    except Exception as error:
        error.add_note('Independent saved-result audit function: aggregate')
        raise
ARMS = ('A14', 'B26', 'C26_static')
CONDITIONS = ('normal', 'static_mid', 'shuffled', 'reversed')

def order_for(length: int, sid: str, condition: str) -> np.ndarray:
    """Reconstruct each temporal control with a local independent generator."""
    try:
        if condition == 'normal':
            return np.arange(length, dtype=np.int64)
        if condition == 'static_mid':
            return np.repeat(np.int64(length // 2), length)
        if condition == 'reversed':
            return np.arange(length - 1, -1, -1, dtype=np.int64)
        check(condition == 'shuffled', 'Unknown control')
        seed = int(hashlib.sha256(('stage5i/' + sid).encode()).hexdigest()[:16], 16)
        return np.random.default_rng(seed).permutation(length).astype(np.int64)
    except Exception as error:
        error.add_note('Independent saved-result audit function: order_for')
        raise

def run() -> dict[str, Any]:
    """Audit allowed sources, all optimizer inputs, saved predictions and immutable history."""
    try:
        plan = read(BASE / 'plan.json')
        check(plan['steps'] == 1820 and plan['seeds'] == [42, 43] and (plan['evaluations'] == [910, 1820]), 'Fixed scope')
        data = read(bound(plan['data_plan']))
        bound(plan['policy'])
        for binding in plan['bindings']:
            bound(binding)
        previous = read(ROOT / 'experiments/offline_confirmed_learning_step05h_20261019/data_plan.json')
        source3 = read(ROOT / 'experiments/offline_dataset_step03_20261013/results/cache_handoff.json')
        original = {row['clip_id']: row for row in previous['records']}
        original.update({row['clip_id']: row for row in source3['records'] if row['kind'] == 'full_sentence'})
        expected_cohorts = {'train_phrase14': set(previous['arms']['B14']), 'train_sentence12': {r['clip_id'] for r in source3['records'] if r['split'] == 'train' and r['kind'] == 'full_sentence'}, 'dev_phrase3': set(previous['dev']), 'dev_sentence3': {r['clip_id'] for r in source3['records'] if r['split'] == 'dev' and r['kind'] == 'full_sentence'}}
        check({k: set(v) for k, v in data['cohorts'].items()} == expected_cohorts, 'Independent cohort membership')
        check([len(expected_cohorts[k]) for k in expected_cohorts] == [14, 12, 3, 3], 'Cohort counts')
        rows = {row['clip_id']: row for row in data['records']}
        check(len(rows) == len(data['records']) == 32 and rows.keys() == original.keys(), 'Source IDs')
        check(data['experimental_split_overrides'] == {}, 'Unexpected split override')
        expected_train = expected_cohorts['train_phrase14'] | expected_cohorts['train_sentence12']
        expected_dev = expected_cohorts['dev_phrase3'] | expected_cohorts['dev_sentence3']
        check(set(data['arms']) == set(ARMS) and set(data['arms']['A14']) == expected_cohorts['train_phrase14'], 'A14 allowlist')
        check(set(data['arms']['B26']) == expected_train and data['arms']['B26'] == data['arms']['C26_static'], 'B/C allowlists')
        check(data['held'] == previous['held'] and {r['review_id'] for r in data['held']} == {f'Q{i:02}' for i in range(3, 9)}, 'Held scope')
        check(not {r['parent_id'] for r in data['held']} & {r['parent_id'] for r in rows.values()}, 'Held parent entered')
        check(set(data['held_parent_ids']) == {r['parent_id'] for r in data['held']}, 'Held parent list')
        for field in ('parent_id', 'recording_group', 'text'):
            check(not {rows[sid][field] for sid in expected_train} & {rows[sid][field] for sid in expected_dev}, 'Train/dev leakage: ' + field)
        for a in expected_train:
            for b in expected_dev:
                ta, tb = (text(rows[a]['text']).replace(' ', ''), text(rows[b]['text']).replace(' ', ''))
                check(ta not in tb and tb not in ta, 'Train/dev contiguous text leakage')
        inputs, labels, metadata, prepared = ({}, {}, {}, {})
        for sid, row in rows.items():
            for field in ('parent_id', 'recording_group', 'speaker_id', 'split', 'kind', 'T', 'text', 'npz_path', 'npz_sha256', 'metadata_path', 'metadata_sha256'):
                check(row[field] == original[sid][field], 'Changed source field: ' + field)
            meta = read(bound({'path': row['metadata_path'], 'sha256': row['metadata_sha256']}))
            metadata[sid] = meta
            source = bound({'path': row['npz_path'], 'sha256': row['npz_sha256'], 'bytes': row['npz_bytes']})
            check(meta['training_ready'] is False and meta['provisional'] is True and (row['training_ready'] is False) and (row['provisional'] is True), 'Source flags')
            if row['kind'] == 'full_sentence':
                evidence = meta['parent_audio_review_provenance']
                check(evidence['parent_transcript_confirmed'] and evidence['parent_audio_boundary_confirmed'], 'Parent audio evidence')
                check(evidence['canonical_parent_text'] == row['text'], 'Parent canonical text')
            with np.load(source, allow_pickle=False) as packed:
                check(set(packed.files) == {'schema', 'frames_rgb', 'length', 'label', 'metadata_json', 'training_ready', 'provisional'}, 'Visual-only NPZ payload')
                check(json.loads(packed['metadata_json'].item()) == meta, 'NPZ sidecar metadata')
                rgb = packed['frames_rgb']
                check(array_digest(rgb) == meta['rgb_uint8_sha256'], 'Source RGB hash')
                check(packed['training_ready'].item() is False and packed['provisional'].item() is True, 'Packed source flags')
                check(rgb.dtype == np.uint8 and rgb.shape == (row['T'], 96, 96, 3) and (int(packed['length']) == row['T']), 'Source shape')
                labels[sid] = packed['label'].copy()
                check(labels[sid].tolist() == meta['label'] and labels[sid].shape == (row['label_length'],), 'Source target')
                check(len(labels[sid]) + int(np.sum(labels[sid][1:] == labels[sid][:-1])) <= row['T'], 'Target CTC feasibility')
                pixels = rgb.astype(np.float32) / np.float32(255)
                pixels -= np.array([0.485, 0.456, 0.406], np.float32)
                pixels /= np.array([0.229, 0.224, 0.225], np.float32)
                inputs[sid] = np.ascontiguousarray(np.moveaxis(pixels, -1, 0))
                check(np.isfinite(inputs[sid]).all(), 'Normalized finite')
            for condition in CONDITIONS:
                indices = order_for(row['T'], sid, condition)
                transformed = np.ascontiguousarray(inputs[sid][:, indices])
                prepared[sid, condition] = {'clip_id': sid, 'condition': condition, 'frame_indices': indices.tolist(), 'input_sha256': array_digest(transformed)}
        for member in ('video', 'label'):
            check(not {metadata[s]['source_members'][member]['sha256'] for s in expected_train} & {metadata[s]['source_members'][member]['sha256'] for s in expected_dev}, 'Train/dev source hash leakage')
        for binding in data['source_handoffs']:
            bound(binding)
        batch_hashes = {}

        def batch_hash(ids: list[str], condition: str) -> str:
            """Rebuild the actual padded two-example tensor and memoize its digest."""
            try:
                key = (tuple(ids), condition)
                if key not in batch_hashes:
                    max_length = max((rows[sid]['T'] for sid in ids))
                    padded = np.empty((2, 3, max_length, 96, 96), np.float32)
                    for index, sid in enumerate(ids):
                        length = rows[sid]['T']
                        source_indices = np.array(prepared[sid, condition]['frame_indices'], np.int64)
                        with_padding = source_indices[np.minimum(np.arange(max_length), length - 1)]
                        padded[index] = inputs[sid][:, with_padding]
                    batch_hashes[key] = array_digest(padded)
                return batch_hashes[key]
            except Exception as error:
                error.add_note('Independent saved-result audit function: batch_hash')
                raise
        initial, summaries, schedule_rows = ({}, {}, {})
        posterior_count = frame_count = update_count = checkpoints = 0
        for seed in [42, 43]:
            for arm in ARMS:
                name = f'{arm}_seed{seed}'
                directory = BASE / 'results' / name
                result, contract = (read(directory / 'result.json'), read(directory / 'run_contract.json'))
                check(result['status'] == 'completed' and result['physical_updates'] == result['total_history'] == 1820, name + ' completion')
                check(contract['resume'] is None, 'This audit expects six fresh continuous runs')
                check(result['identity']['plan_sha256'] == digest(BASE / 'plan.json'), 'Plan identity')
                check(contract['optimizer_ids'] == data['arms'][arm], 'Optimizer IDs')
                condition_train = 'static_mid' if arm == 'C26_static' else 'normal'
                check(contract['training_condition'] == plan['training_conditions'][arm] == condition_train, 'Training condition')
                initial[name] = result['initial_state_hashes']
                check(initial[name] == contract['initial_state_hashes'], 'Initial state record')
                tasks = read(bound(plan['schedules'][arm][str(seed)]))['updates']
                schedule_rows[name] = tasks
                logs = [json.loads(line) for line in (directory / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
                check(len(tasks) == len(logs) == 1820, 'Schedule/progress length')
                for update, (task, log) in enumerate(zip(tasks, logs), 1):
                    check(log['update'] == task['update'] == update and log['clip_ids'] == task['clip_ids'], 'Progress order')
                    lr = 0.001 if update <= 910 else 0.0003 if update <= 1365 else 0.0001
                    lengths = [rows[sid]['T'] for sid in task['clip_ids']]
                    check(log['learning_rate'] == task['lr'] == lr and log['lengths'] == task['lengths'] == lengths, 'Progress LR/length')
                    check(log['optimizer_updated'] and np.isfinite(log['loss']), 'Progress update/loss')
                    check(log['training_condition'] == condition_train, 'Progress training condition')
                    check(log['input_audits'] == [prepared[sid, condition_train] for sid in task['clip_ids']], 'Actual unpadded optimizer inputs')
                    check(log['padded_batch_sha256'] == batch_hash(task['clip_ids'], condition_train), 'Actual padded optimizer input')
                    check(log['true_frames'] == task['true_frames'] == sum(lengths), 'Progress true frames')
                    check(log['dense_frames'] == task['dense_frames'] == 2 * max(lengths), 'Progress dense frames')
                    check(set(log['clip_ids']) <= set(data['arms'][arm]), 'Nontrain optimizer input')
                    check(np.isfinite(log['gradient_norm_before_clip']) and np.isfinite(log['gradient_norm_after_clip']), 'Finite gradients')
                exposure = Counter((sid for task in tasks for sid in task['clip_ids']))
                check(set(exposure) == set(data['arms'][arm]) and set(exposure.values()) == {260 if arm == 'A14' else 140}, 'Exposure balance')
                check(dict(exposure) == plan['planned_accounting'][name]['exposures'], 'Planned exposures')
                for field in ('true_frames', 'dense_frames'):
                    check(sum((t[field] for t in tasks)) == result[field] == plan['planned_accounting'][name][field], 'Final ' + field)
                check(result['seconds'] <= plan['limits']['wall_seconds_per_run'], 'Run wall limit')
                for field in ('audio_inputs', 'teacher_calls', 'service_changes'):
                    check(result[field] == 0, 'Unexpected side effect: ' + field)
                check(result['source_approval_changed'] is False and result['source_training_ready'] is False, 'Approval promotion')
                check(len(result['checkpoints']) == 2, 'Checkpoint count')
                for binding in result['checkpoints']:
                    bound(binding)
                    checkpoints += 1
                check(len(result['evaluations']) == 2, 'Evaluation checkpoints')
                summaries[name] = {}
                for update, link in zip([910, 1820], result['evaluations']):
                    summary_path = bound(link['summary'])
                    summary = read(summary_path)
                    eval_dir = summary_path.parent
                    stored = [json.loads(line) for line in (eval_dir / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
                    check(stored == summary['rows'], 'Summary/JSONL rows')
                    isolation = read(eval_dir / 'isolation.json')
                    for field in ('state', 'gradients', 'modes'):
                        check(isolation[field + '_before'] == isolation[field + '_after'], 'Isolation ' + field)
                    wanted_ids = data['arms'][arm] if update == 910 else sorted(rows)
                    wanted_conditions = ('normal',) if update == 910 else CONDITIONS
                    expected = {(sid, condition) for sid in wanted_ids for condition in wanted_conditions}
                    actual = {(r['clip_id'], r['condition']) for r in stored}
                    check(len(stored) == len(actual) and actual == expected, 'Evaluation missing/duplicate pairs')
                    scores = defaultdict(list)
                    for row in stored:
                        sid, condition = (row['clip_id'], row['condition'])
                        source, length = (rows[sid], rows[sid]['T'])
                        check(row['status'] == 'success' and row['audio_input'] is False, 'Evaluation status/input')
                        check(row['arm'] == arm and row['seed'] == seed and (row['update'] == update), 'Evaluation identity')
                        check(row['reference'] == source['text'] and row['reference_label_ids'] == labels[sid].tolist(), 'Target identity')
                        check(row['seen_in_optimizer'] == (sid in data['arms'][arm]), 'Training exposure annotation')
                        expected_cohort = next((k for k, values in expected_cohorts.items() if sid in values))
                        check(row['cohort'] == expected_cohort, 'Cohort label')
                        indices = order_for(length, sid, condition)
                        check(row['frame_indices'] == prepared[sid, condition]['frame_indices'], 'Temporal control')
                        check(row['input_sha256'] == prepared[sid, condition]['input_sha256'], 'Actual transformed input hash')
                        check(row['input_audit']['normalized_input_sha256'] == array_digest(inputs[sid]), 'Normalized input hash')
                        with np.load(bound(row['posterior']), allow_pickle=False) as packed:
                            values = packed['log_probs']
                            check(values.dtype == np.float32 and values.shape == (length, 69) and np.isfinite(values).all(), 'Posterior shape/finite')
                            check(np.allclose(np.exp(values.astype(np.float64)).sum(1), 1, atol=1e-05, rtol=1e-05), 'Posterior normalization')
                            check(array_digest(values) == row['posterior_array_sha256'], 'Posterior hash')
                            check(np.array_equal(packed['frame_indices'], indices) and np.array_equal(packed['reference_labels'], labels[sid]), 'Posterior metadata')
                            argmax = values.argmax(1).tolist()
                            check(argmax == packed['raw_argmax_ids'].tolist() == row['raw_argmax_ids'], 'Argmax')
                        collapsed, prediction = decode(argmax)
                        check(collapsed == row['collapsed_ids'] and prediction == row['raw_text'] and (text(prediction) == row['metric_text']), 'CTC/Hangul/UFFFD decoding')
                        independent = score(source['text'], prediction)
                        check(independent == row['metrics'], 'Independent edit counts')
                        scores[f'{expected_cohort}/{condition}'].append(independent)
                        posterior_count += 1
                        frame_count += length
                    recalculated = {key: aggregate(values) for key, values in scores.items()}
                    check(recalculated == summary['cohorts'] == link['cohorts'], 'Aggregate metrics')
                    check(summary['count'] == link['count'] == len(stored) and summary['failures'] == link['failures'] == 0, 'Summary counts')
                    check(summary['frames'] == link['frames'] == sum((r['T'] for r in stored)), 'Evaluation frame accounting')
                    summaries[name][str(update)] = recalculated
                update_count += len(logs)
            check(initial[f'A14_seed{seed}'] == initial[f'B26_seed{seed}'] == initial[f'C26_static_seed{seed}'], 'A/B/C paired initial state')
            check(schedule_rows[f'B26_seed{seed}'] == schedule_rows[f'C26_static_seed{seed}'], 'Identical B/C batches')
            for epoch in range(1, 141):
                a = [s for row in schedule_rows[f'A14_seed{seed}'] if row['epoch'] == epoch for s in row['clip_ids']]
                b = [s for row in schedule_rows[f'B26_seed{seed}'] if row['epoch'] == epoch for s in row['clip_ids'] if s in data['arms']['A14']]
                check(a == b, 'Shared old-parent order')
        check(update_count == plan['main_updates'] == 10920 and posterior_count == plan['evaluation']['planned_rows'] == 900 and (checkpoints == 12), 'Total accounting')
        preflight = read(BASE / 'results/preflight/result.json')
        check(preflight['status'] == 'passed' and preflight['physical_updates'] == 5 and preflight['state_exact'] and preflight['next_update_exact'], 'Preflight record')
        bound(preflight['checkpoint'])
        check(preflight['reference_final_hashes'] == preflight['restored_final_hashes'], 'Preflight final hashes')
        preserved = {}
        prior = read(ROOT / 'experiments/offline_phrase_review_step05f_20261018/audit/source_preservation_final.json')
        manifests = {'scope': bound(prior['service_and_prior_scope_binding']), 'review_r8': ROOT / 'experiments/offline_phrase_boundary_step05g_20261018/results/review_snapshot_manifest_r8.json', 'stage5h': bound(data['frozen_previous'])}
        for name, path in manifests.items():
            entries = read(path)['artifacts']
            for entry in entries:
                if entry.get('exists', True):
                    bound(entry)
                else:
                    check(not Path(entry['path']).exists(), 'Previously absent scope artifact appeared')
            preserved[name] = len(entries)
        return {'schema': 'offline-stage5i-independent-saved-results-audit-v1', 'status': 'passed', 'client_date': '2026-10-19', 'created_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': digest(BASE / 'plan.json'), 'auditor_sha256': digest(__file__), 'cache_inputs_checked': 32, 'posteriors_checked': posterior_count, 'posterior_frames': frame_count, 'main_updates_checked': update_count, 'preflight_updates_recorded': 5, 'total_physical_updates_recorded': update_count + 5, 'checkpoint_file_hashes_checked': checkpoints, 'preflight_checkpoint_file_hashes_checked': 1, 'optimizer_transformed_inputs_checked': update_count * 2, 'optimizer_padded_batches_checked': update_count, 'distinct_padded_batch_hashes': len(batch_hashes), 'paired_initial_states_equal': True, 'shared_order_epochs_per_seed': 140, 'B_C_schedules_identical': True, 'preservation_counts': preserved, 'cohorts': summaries, 'decoder': 'independent numeric Hangul composition with U+FFFD preservation', 'metrics': 'independent full-matrix Levenshtein, NFC outer-strip, corpus totals', 'network_forward_calls': 0, 'checkpoint_tensor_loads': 0, 'original_media_reads': 0, 'limitations': 'Saved hashes and outputs validated; no independent neural rerun or checkpoint-tensor reconstruction. Dev phrase/full share3parents and are not six independent examples.'}
    except Exception as error:
        error.add_note('Independent saved-result audit function: run')
        raise
if __name__ == '__main__':
    target = BASE / 'audit/final_saved_results_check.json'
    try:
        check(not target.exists(), 'Audit already exists')
        report = run()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'cohorts'}, ensure_ascii=False))
    except Exception:
        target.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        failure = target.parent / f'saved_audit_failure_{stamp}.json'
        with failure.open('x', encoding='utf-8') as stream:
            json.dump({'status': 'audit_failed', 'auditor_sha256': digest(__file__), 'error': traceback.format_exc(), 'model_or_training_mutated': False}, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        raise
