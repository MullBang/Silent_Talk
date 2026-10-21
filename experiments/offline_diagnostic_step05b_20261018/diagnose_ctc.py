"""Post-hoc CPU analysis of saved probabilities; no model or optimizer imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/offline_comparison_step05_20261013'
MANIFEST = SOURCE / 'results/completion_manifest.json'
MANIFEST_SHA = '30d82e9ee65631ed1ec4e11961d7d45610ec13ac21c6f3708c0539a947fb9807'
PLAN = HERE / 'diagnostic_plan.json'
TIE_TOLERANCE = 1e-9


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def check(ref):
    path = Path(ref['path'])
    assert path.is_file() and sha(path) == ref['sha256'], str(path)
    if 'bytes' in ref:
        assert path.stat().st_size == ref['bytes'], str(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def minimum_frames(labels):
    return len(labels) + sum(a == b for a, b in zip(labels[:-1], labels[1:]))


def ctc_nll(log_probs, labels):
    """Negative log sum of all CTC alignments. Infinity means impossible length."""
    lp = np.asarray(log_probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if lp.ndim != 2 or lp.shape[1] < 2 or not np.isfinite(lp).all():
        raise ValueError('Expected finite TxV log probabilities')
    if y.ndim != 1 or np.any(y <= 0) or np.any(y >= lp.shape[1]):
        raise ValueError('Targets must be nonblank IDs in the vocabulary')
    T = len(lp)
    if T < minimum_frames(y):
        return float('inf')
    if len(y) == 0:
        return -float(lp[:, 0].sum())
    states = np.zeros(2 * len(y) + 1, dtype=np.int64)
    states[1::2] = y
    previous = np.full(len(states), -np.inf, dtype=np.float64)
    previous[:2] = lp[0, states[:2]]
    can_skip = np.zeros(len(states), dtype=bool)
    can_skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    for t in range(1, T):
        one = np.concatenate(([-np.inf], previous[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        two[~can_skip] = -np.inf
        previous = np.logaddexp(np.logaddexp(previous, one), two) + lp[t, states]
    return -float(np.logaddexp(previous[-1], previous[-2]))


def collapse(ids):
    return [int(value) for index, value in enumerate(ids)
            if value != 0 and (index == 0 or value != ids[index - 1])]


def prepare():
    assert sha(MANIFEST) == MANIFEST_SHA
    frozen = {item['path']: item for item in read(MANIFEST)['artifacts']}
    paths = sorted(SOURCE.glob('runs/**/evaluation_summary.json'))
    assert len(paths) == 12
    identities = set()
    canonical = {}
    normal_count = 0
    for path in paths:
        check(frozen[str(path.resolve())])
        result = read(path)
        identity = result['identity']
        identities.add((identity['seed'], identity['arm'], identity['completed_update']))
        assert result['failures'] == 0 and len(result['rows']) == 90
        for row in result['rows']:
            if row['condition'] != 'normal':
                continue
            normal_count += 1
            record = {key: row[key] for key in ['clip_id', 'parent_id', 'kind', 'split', 'T']}
            record.update(text=row['reference_text'], labels=row['reference_label_ids'])
            record['speaker_id'] = record['parent_id'].split('_')[5]
            # parent spelling lip_J_1_F_02_C004_A_011_014 has C004 at index5.
            assert record['speaker_id'].startswith('C')
            if row['clip_id'] in canonical:
                assert canonical[row['clip_id']] == record
            canonical[row['clip_id']] = record
    assert normal_count == 360 and len(canonical) == 30 and len(identities) == 12
    pools = {}
    for kind in ['phrase', 'full_sentence']:
        grouped = {}
        for record in canonical.values():
            if record['kind'] != kind or record['split'] != 'train':
                continue
            key = tuple(record['labels'])
            if key not in grouped:
                grouped[key] = {'text': record['text'], 'labels': record['labels'], 'source_clip_ids': [], 'speakers': []}
            assert grouped[key]['text'] == record['text']
            grouped[key]['source_clip_ids'].append(record['clip_id'])
            grouped[key]['speakers'].append(record['speaker_id'])
        pools[kind] = []
        for index, record in enumerate(sorted(grouped.values(), key=lambda x: x['source_clip_ids'][0])):
            record['candidate_id'] = kind + '_train_' + str(index + 1).zfill(2)
            record['speakers'] = sorted(set(record['speakers']))
            record['source_clip_ids'].sort()
            pools[kind].append(record)
    plan = {
        'schema': 'offline-stage5b-saved-posterior-diagnostic-plan-v1',
        'client_date': '2026-10-18',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Post-hoc descriptive diagnosis only; fixed prior primary result is unchanged',
        'source_manifest': binding(MANIFEST), 'code': binding(__file__),
        'tests': binding(HERE / 'test_diagnose_ctc.py'),
        'evaluations': [binding(path) for path in paths],
        'records': sorted(canonical.values(), key=lambda row: row['clip_id']),
        'candidate_pools': pools,
        'selection': 'All360 normal rows: 4runs x 3checkpoints x 30clips; no cherry-picking',
        'scoring': 'NumPy float64 CTC sum over alignments using stored float32 log probabilities without renormalization; blank0',
        'candidate_policy': 'Unique same-kind train label sequences plus reference if missing. Actual greedy collapsed token sequence scored separately.',
        'rank_policy': '1 + number of candidates with NLL smaller by >1e-9; ties reported; impossible sequences retained with null NLL/status',
        'token_normalized_score': 'NLL / number of target tokens, diagnostic heuristic not a sequence probability; empty greedy normalization null',
        'tie_tolerance': TIE_TOLERANCE,
        'no_performance_claim': 'Reference-inclusive ranking is not open-vocabulary accuracy. Unequal length/feasibility/speaker candidate counts confound preference.',
        'new_neural_calls': 0, 'new_optimizer_updates': 0, 'checkpoint_tensor_loads': 0,
        'source_media_reads': 0, 'new_downloads': 0, 'service_changes': 0,
        'training_ready_promotions': 0,
    }
    save(PLAN, plan)
    print(json.dumps({'status': 'prepared', 'plan': binding(PLAN),
                      'candidate_counts': {key: len(value) for key, value in pools.items()}}))


def score_record(lp, labels):
    nll = ctc_nll(lp, labels)
    feasible = minimum_frames(labels) <= len(lp)
    assert bool(np.isfinite(nll)) == feasible
    return {'labels': list(labels), 'label_length': len(labels),
            'minimum_frames': minimum_frames(labels), 'feasible': feasible,
            'status': 'finite' if feasible else 'impossible_ctc_length',
            'nll': nll if feasible else None,
            'nll_per_token': nll / len(labels) if feasible and len(labels) else None}


def ranking(candidates, reference, field):
    value = reference[field]
    assert value is not None
    finite = [candidate for candidate in candidates if candidate[field] is not None]
    minimum = min(candidate[field] for candidate in finite)
    return {'reference_rank': 1 + sum(candidate[field] < value - TIE_TOLERANCE for candidate in finite),
            'reference_tie_count': sum(abs(candidate[field] - value) <= TIE_TOLERANCE for candidate in finite),
            'top_candidate_ids': [candidate['candidate_id'] for candidate in finite
                                  if abs(candidate[field] - minimum) <= TIE_TOLERANCE],
            'reference_minus_best': value - minimum,
            'finite_candidates': len(finite), 'total_candidates': len(candidates)}


def run():
    plan = read(PLAN)
    check(plan['source_manifest']); check(plan['code']); check(plan['tests'])
    records = {record['clip_id']: record for record in plan['records']}
    rows = []
    calls = 0
    for source in plan['evaluations']:
        check(source)
        evaluation = read(source['path'])
        for row in evaluation['rows']:
            if row['condition'] != 'normal':
                continue
            check(row['posterior'])
            with np.load(row['posterior']['path'], allow_pickle=False) as payload:
                lp = payload['log_probs'].astype(np.float64)
                raw = lp.argmax(axis=1).tolist()
                greedy_ids = collapse(raw)
                assert raw == row['raw_argmax_ids'] == payload['raw_argmax_ids'].tolist()
                assert greedy_ids == row['collapsed_ids'] == payload['collapsed_ids'].tolist()
            assert lp.shape == (row['T'], 69)
            sum_error = float(np.abs(np.exp(lp).sum(axis=1) - 1).max())
            assert sum_error < 1e-5
            record = records[row['clip_id']]
            candidates = []
            for candidate in plan['candidate_pools'][row['kind']]:
                candidates.append({**candidate, **score_record(lp, candidate['labels']),
                                   'is_reference': candidate['labels'] == record['labels'], 'in_training_pool': True})
            if not any(candidate['is_reference'] for candidate in candidates):
                candidates.append({'candidate_id': 'reference', 'text': record['text'],
                                   'source_clip_ids': [row['clip_id']], 'speakers': [record['speaker_id']],
                                   **score_record(lp, record['labels']), 'is_reference': True, 'in_training_pool': False})
            reference = next(candidate for candidate in candidates if candidate['is_reference'])
            assert reference['feasible']
            greedy = score_record(lp, greedy_ids)
            calls += len(candidates) + 1
            rows.append({
                'seed': row['identity']['seed'], 'arm': row['identity']['arm'],
                'update': row['identity']['completed_update'],
                **{key: record[key] for key in ['clip_id', 'parent_id', 'kind', 'split', 'T', 'speaker_id']},
                'reference_text': row['reference_text'], 'greedy_text': row['raw_text'],
                'greedy_exact': row['metrics']['exact'], 'saved_metrics': row['metrics'],
                'source_evaluation': source, 'posterior': row['posterior'],
                'log_probability_sum_max_error': sum_error,
                'reference_label_length': len(record['labels']),
                'reference_ctc_slack': row['T'] - minimum_frames(record['labels']),
                'candidates': candidates, 'greedy': greedy,
                'reference_nll': reference['nll'],
                'reference_minus_greedy_nll': reference['nll'] - greedy['nll'],
                'total_score_ranking': ranking(candidates, reference, 'nll'),
                'per_token_ranking': ranking(candidates, reference, 'nll_per_token'),
            })
    assert len(rows) == 360
    groups = []
    for key in sorted({(r['seed'], r['arm'], r['update'], r['split'], r['kind']) for r in rows}):
        group = [r for r in rows if (r['seed'], r['arm'], r['update'], r['split'], r['kind']) == key]
        groups.append({
            **dict(zip(['seed', 'arm', 'update', 'split', 'kind'], key)), 'count': len(group),
            'greedy_exact': sum(r['greedy_exact'] for r in group),
            'reference_top_total': sum(r['total_score_ranking']['reference_rank'] == 1 for r in group),
            'reference_top_per_token': sum(r['per_token_ranking']['reference_rank'] == 1 for r in group),
            'reference_total_better_than_greedy': sum(r['reference_minus_greedy_nll'] < -TIE_TOLERANCE for r in group),
            'reference_total_tied_with_greedy': sum(abs(r['reference_minus_greedy_nll']) <= TIE_TOLERANCE for r in group),
            'reference_total_worse_than_greedy': sum(r['reference_minus_greedy_nll'] > TIE_TOLERANCE for r in group),
            'median_reference_minus_greedy_nll': float(np.median([r['reference_minus_greedy_nll'] for r in group])),
            'impossible_candidates': sum(not c['feasible'] for r in group for c in r['candidates']),
            'total_candidate_pairs': sum(len(r['candidates']) for r in group),
        })
    output = {'schema': 'offline-stage5b-saved-ctc-diagnostic-v1', 'status': 'completed',
              'plan': binding(PLAN), 'rows': rows, 'groups': groups,
              'posterior_rows': len(rows), 'posterior_frames': sum(r['T'] for r in rows),
              'candidate_pairs': sum(len(r['candidates']) for r in rows),
              'greedy_pairs': len(rows), 'ctc_computations': calls,
              'impossible_candidate_pairs': sum(not c['feasible'] for r in rows for c in r['candidates']),
              'new_neural_calls': 0, 'new_optimizer_updates': 0, 'checkpoint_tensor_loads': 0,
              'source_media_reads': 0, 'service_changes': 0,
              'limitations': [plan['no_performance_claim'], plan['scope']]}
    save(HERE / 'results/ctc_candidates.json', output)
    save(HERE / 'results/ctc_summary.json', {key: value for key, value in output.items() if key != 'rows'})
    print(json.dumps({key: output[key] for key in ['status', 'posterior_rows', 'posterior_frames', 'candidate_pairs',
                                                 'greedy_pairs', 'ctc_computations', 'impossible_candidate_pairs']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
