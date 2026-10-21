"""CPU-only CTC candidate diagnosis from immutable, already saved posteriors.

Reference-inclusive candidate ranks are diagnostic oracle information, never
deployable accuracy. Per-target-token scores are a heuristic, not probability.
No model architecture, checkpoint, visual input, or CUDA API is loaded here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
EPSILON = 1e-8
SCORE_KEYS = ('total_log_probability', 'mean_log_probability_per_target_token')
NORMALIZATION_ATOL = 1e-5
PURE_FILES = {
    'tokenizer': (ROOT / 'backend/models/tokenizer.py', 'c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714'),
    'diagnostics': (ROOT / 'experiments/olkavs_official/diagnostic_utils.py', '3bac7121c89a8cfbae12b97dc4cfbdc8147ca44eccc2eb9d0af5c69f234d1ef9'),
    'metrics': (ROOT / 'experiments/olkavs_official/evaluation_metrics.py', 'eb7fc30b04e08143d6b7f455f1ad7918681c42d395f4da58b1ced774a502b4b3'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load_pure(name):
    path, digest = PURE_FILES[name]
    require(bind(path)['sha256'] == digest, 'Frozen pure helper changed: ' + name)
    spec = importlib.util.spec_from_file_location('step14_pure_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tokenizer, diagnostics = load_pure('tokenizer'), load_pure('diagnostics')
_old_diagnostics = sys.modules.get('diagnostic_utils')
sys.modules['diagnostic_utils'] = diagnostics
try:
    metrics = load_pure('metrics')
finally:
    if _old_diagnostics is None:
        del sys.modules['diagnostic_utils']
    else:
        sys.modules['diagnostic_utils'] = _old_diagnostics


def scoring_code():
    return [bind(path) for path in [Path(__file__), BASE / 'test_scoring.py', *[value[0] for value in PURE_FILES.values()]]]


def minimum_frames(tokens):
    require(isinstance(tokens, list) and all(type(t) is int and 1 <= t <= 68 for t in tokens), 'Invalid CTC target token')
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def validate_posterior(log_probs):
    require(isinstance(log_probs, np.ndarray) and log_probs.dtype == np.float32
            and log_probs.ndim == 2 and log_probs.shape[1] == 69 and len(log_probs) > 0,
            'Expected FP32 T-by-69 saved log probabilities')
    require(np.isfinite(log_probs).all(), 'Nonfinite saved posterior')
    lp64 = torch.from_numpy(log_probs.astype(np.float64, copy=True))
    error = float(torch.logsumexp(lp64, dim=-1).abs().max())
    require(error <= NORMALIZATION_ATOL, 'Saved frame log probabilities are not normalized')
    return lp64, error


def ctc_score(lp64, tokens):
    require(isinstance(lp64, torch.Tensor) and lp64.device.type == 'cpu' and lp64.dtype == torch.float64
            and lp64.ndim == 2 and lp64.shape[1] == 69 and len(lp64) > 0 and torch.isfinite(lp64).all().item(),
            'Scoring requires finite Float64 CPU log probabilities')
    min_frames = minimum_frames(tokens)
    base = {'tokens': list(tokens), 'target_tokens': len(tokens), 'minimum_frames': min_frames}
    if min_frames > len(lp64):
        return {**base, 'status': 'infeasible', 'total_log_probability': None,
                'mean_log_probability_per_target_token': None, 'probability_is_zero': True}
    if not tokens:
        total = float(lp64[:, 0].sum())
    else:
        loss = F.ctc_loss(lp64.unsqueeze(1), torch.tensor(tokens, dtype=torch.long),
                          torch.tensor([len(lp64)]), torch.tensor([len(tokens)]),
                          blank=0, reduction='none', zero_infinity=False)
        total = -float(loss[0])
    require(np.isfinite(total), 'Nonfinite CTC score for feasible target')
    return {**base, 'status': 'ok', 'total_log_probability': total,
            'mean_log_probability_per_target_token': total / len(tokens) if tokens else None,
            'probability_is_zero': False}


def validate_source(source, lp64, normalization_error):
    row = source['source_record']
    require(canonical_sha(row) == source['source_record_sha256'], 'Source record checksum changed')
    require(row['status'] == 'ok' and row['error'] is None, 'Original source inference failed')
    require(row['id'] == source['id'] and row['speaker_id'] == source['speaker_id']
            and row['condition'] == source['condition'] and row['reference'] == source['text']
            and row['frames'] == source['frames'] == len(lp64)
            and row['target_token_ids'] == source['tokens'] == tokenizer.encode_text(source['text'])
            and row['source_split'] == 'train', 'Source row provenance changed')
    require(len(source['tokens']) > 0 and minimum_frames(source['tokens']) <= len(lp64), 'Reference must be CTC-feasible')
    raw = lp64.argmax(-1).tolist()
    collapsed = diagnostics.collapse_ctc(raw, blank_id=0)
    prediction = tokenizer.compose_jamos([tokenizer.IDX_TO_GRAPHEME[i] for i in collapsed], strict=False)
    require(raw == row['raw_argmax_ids'] and collapsed == row['collapsed_ids']
            and prediction == row['prediction'] and prediction.count('\ufffd') == row['replacement_characters'],
            'Saved greedy decoding changed')
    try:
        strict = {'status': 'ok', 'prediction': tokenizer.decode_labels(collapsed), 'error': None}
    except ValueError as exc:
        strict = {'status': 'failed', 'prediction': None, 'error': f'ValueError: {exc}'}
    require(strict == row['strict_label_decoder_diagnostic'], 'Saved strict decoder diagnostic changed')
    require(row['metrics'] == metrics.counts(source['text'], prediction), 'Saved greedy text metrics changed')
    token_metrics = {'token_edits': diagnostics.edit_distance(source['tokens'], collapsed),
                     'target_tokens': len(source['tokens']), 'exact': source['tokens'] == collapsed}
    require(row['token_metrics'] == token_metrics, 'Saved greedy token metrics changed')
    ctc32 = float(F.ctc_loss(lp64.float().unsqueeze(1), torch.tensor(source['tokens'], dtype=torch.long),
                           torch.tensor([len(lp64)]), torch.tensor([len(source['tokens'])]),
                           blank=0, reduction='mean', zero_infinity=False))
    require(np.isfinite(row['ctc_loss_mean']) and np.isclose(ctc32, row['ctc_loss_mean'], rtol=1e-6, atol=1e-5),
            'Saved reference FP32 CTC changed')
    return {'posterior_shape': list(lp64.shape), 'posterior_saved_dtype': 'float32',
            'normalization_max_abs_logsumexp': normalization_error,
            'raw_argmax_ids': raw, 'collapsed_ids': collapsed, 'prediction': prediction,
            'replacement_characters': prediction.count('\ufffd'), 'metrics': row['metrics'],
            'token_metrics': token_metrics, 'source_ctc_loss_mean_float32': row['ctc_loss_mean'],
            'recomputed_ctc_loss_mean_float32': ctc32, 'raw_collapsed_text_and_metrics_unchanged': True}


def rank_candidates(candidates, score_key):
    require(score_key in SCORE_KEYS, 'Unknown score definition')
    require(len({c['id'] for c in candidates}) == len(candidates), 'Duplicate candidate IDs')
    refs = [c for c in candidates if c['kind'] == 'reference']
    require(len(refs) == 1 and refs[0]['status'] == 'ok', 'Exactly one feasible reference required')
    feasible = [c for c in candidates if c['status'] == 'ok']
    require(all(isinstance(c[score_key], (int, float)) and np.isfinite(c[score_key]) for c in feasible), 'Invalid candidate score')
    require(all(c['status'] in ('ok', 'infeasible') for c in candidates), 'Unrankable source-failed candidate')
    ordered = sorted(feasible, key=lambda c: (-c[score_key], c['id']))
    reference = refs[0]
    score = reference[score_key]
    training = [c for c in feasible if c['kind'] == 'training']

    def best(rows):
        if not rows:
            return None, []
        maximum = max(c[score_key] for c in rows)
        return maximum, sorted(c['id'] for c in rows if abs(c[score_key] - maximum) <= EPSILON)

    top_score, top_ids = best(feasible)
    best_training, best_ids = best(training)
    own_score, own_ids = best([c for c in training if c['same_speaker']])
    other_score, other_ids = best([c for c in training if not c['same_speaker']])
    best_kinds = {c['same_speaker'] for c in training if c['id'] in best_ids}
    relation = 'mixed_tie' if len(best_kinds) == 2 else ('same_speaker' if True in best_kinds else 'other_speaker') if best_kinds else None
    return {'reference_rank': 1 + sum(c[score_key] > score + EPSILON for c in feasible),
            'reference_tie_ids': sorted(c['id'] for c in feasible if abs(c[score_key] - score) <= EPSILON),
            'reference_score': score, 'top_score': top_score, 'top_candidate_ids': top_ids,
            'best_training_score': best_training, 'best_training_ids': best_ids,
            'reference_minus_best_training': score - best_training if best_training is not None else None,
            'best_same_speaker_training_score': own_score, 'best_same_speaker_training_ids': own_ids,
            'best_other_speaker_training_score': other_score, 'best_other_speaker_training_ids': other_ids,
            'same_minus_other_training_score': own_score - other_score if own_score is not None and other_score is not None else None,
            'best_training_relation': relation,
            'ordered_candidate_ids': [c['id'] for c in ordered],
            'infeasible_candidate_ids': sorted(c['id'] for c in candidates if c['status'] == 'infeasible'),
            'feasible_candidates': len(feasible), 'infeasible_candidates': len(candidates) - len(feasible)}


def validate_candidates(source, candidates):
    require(len(candidates) == len({c['id'] for c in candidates}) == 25, 'Expected 25 distinct candidates')
    ref = candidates[0]
    require(ref['kind'] == 'reference' and ref['id'] == source['id'] and ref['tokens'] == source['tokens']
            and ref['text'] == source['text'] and ref['speaker_id'] == source['speaker_id'], 'Changed reference candidate')
    training = candidates[1:]
    require(all(c['kind'] == 'training' for c in training) and [c['id'] for c in training] == sorted(c['id'] for c in training),
            'Training candidates must have fixed stable ID order')
    require(sum(c['same_speaker'] for c in training) == 3, 'Expected three same-speaker and 21 other-speaker training candidates')
    for candidate in candidates:
        require(candidate['tokens'] and candidate['tokens'] == tokenizer.encode_text(candidate['text'])
                and candidate['same_speaker'] == (candidate['speaker_id'] == source['speaker_id']), 'Malformed candidate transcript/tokens/speaker')
        minimum_frames(candidate['tokens'])


def score_source(source, candidates):
    validate_candidates(source, candidates)
    base = {key: source[key] for key in ('seed', 'arm', 'condition', 'cohort', 'id', 'speaker_id', 'frames',
                                        'source_record_sha256', 'source_evaluation', 'posterior_archive', 'posterior_key')}
    base['source_key'] = source['key']
    try:
        lp64, error = validate_posterior(source['log_probs'])
        evidence = validate_source(source, lp64, error)
        scored = [{**candidate, **ctc_score(lp64, candidate['tokens'])} for candidate in candidates]
        rankings = {key: rank_candidates(scored, key) for key in SCORE_KEYS}
        greedy = ctc_score(lp64, evidence['collapsed_ids'])
        require(greedy['status'] == 'ok', 'Argmax-collapsed path must be feasible')
        ref = scored[0]
        evidence['reference_float64_mean_ctc'] = -ref['mean_log_probability_per_target_token']
        evidence['reference_float64_minus_saved_float32_mean_ctc'] = -ref['mean_log_probability_per_target_token'] - source['source_record']['ctc_loss_mean']
        return {**base, 'status': 'ok', 'error': None, 'source_validation': evidence,
                'candidates': scored, 'rankings': rankings, 'greedy_score': greedy,
                'reference_minus_greedy_total_log_probability': ref['total_log_probability'] - greedy['total_log_probability'],
                'candidate_pairs': 25, 'auxiliary_greedy_pairs': 1}
    except Exception as exc:
        return {**base, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'source_validation': None,
                'candidates': [{**candidate, 'target_tokens': len(candidate['tokens']),
                    'minimum_frames': minimum_frames(candidate['tokens']), 'status': 'source_failed',
                    'total_log_probability': None, 'mean_log_probability_per_target_token': None,
                    'probability_is_zero': None} for candidate in candidates], 'rankings': None,
                'greedy_score': {'status': 'source_failed', 'total_log_probability': None,
                                 'mean_log_probability_per_target_token': None},
                'reference_minus_greedy_total_log_probability': None, 'candidate_pairs': 25, 'auxiliary_greedy_pairs': 1}


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['score-all'])
    parser.parse_args()
    import protocol
    policy = protocol.verify_frozen()
    frozen = {ref['path']: ref for ref in policy['execution_code']}
    for ref in scoring_code():
        require(frozen.get(ref['path']) == ref, 'Scoring code not frozen')
    policy_ref = bind(BASE / 'diagnostic_plan.json')
    preflight_path = BASE / 'results/independent_preflight.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'passed' and preflight['experiment_plan'] == policy_ref, 'Passed frozen-policy preflight required')
    output = BASE / 'results/candidate_scores.json'
    contract_path = BASE / 'results/scoring.contract.json'
    require(not output.exists() and not contract_path.exists(), 'Use new immutable scoring output paths')
    torch.set_num_threads(4)
    require(torch.__version__ == policy['runtime']['torch'] and torch.get_num_threads() == policy['runtime']['torch_cpu_threads'] == 4,
            'CPU Torch runtime differs from frozen policy')
    contract = {'schema': 'lipnet-step14-before-scoring-contract-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'experiment_plan': policy_ref, 'independent_preflight': bind(preflight_path),
                'scoring_code': scoring_code(), 'runtime': policy['runtime'], 'new_model_inferences': 0, 'new_optimizer_updates': 0}
    save_new(contract_path, contract)
    rng_before = torch.get_rng_state().clone()
    began = time.perf_counter()
    loaded = protocol.load_sources(policy)
    rows = [score_source(source, protocol.candidates_for(source, loaded['training_candidates'])) for source in loaded['sources']]
    require(len(rows) == len({row['source_key'] for row in rows}) == 144, 'Changed fixed source denominator')
    require(torch.equal(rng_before, torch.get_rng_state()), 'CPU RNG changed during deterministic scoring')
    protocol.verify_frozen()
    value = {'schema': 'lipnet-step14-posterior-candidate-scores-v1', 'status': 'completed',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': policy_ref,
             'before_scoring_contract': bind(contract_path), 'scoring_code': scoring_code(),
             'scoring_contract': policy['scoring_contract'], 'rows': rows,
             'source_rows': 144, 'candidate_pairs': 3600, 'auxiliary_greedy_pairs': 144, 'total_ctc_pairs': 3744,
             'failed_source_rows': sum(row['status'] != 'ok' for row in rows),
             'infeasible_candidate_pairs': sum(c['status'] == 'infeasible' for row in rows for c in row['candidates']),
             'source_failed_candidate_pairs': sum(c['status'] == 'source_failed' for row in rows for c in row['candidates']),
             'new_model_inferences': 0, 'new_optimizer_updates': 0, 'checkpoint_payloads_loaded': 0,
             'visual_input_caches_loaded': 0, 'audio_used': False, 'teacher_used': False,
             'service_weights_updated': False, 'cpu_rng_unchanged': True,
             'elapsed_seconds': time.perf_counter() - began,
             'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'dtype': 'float64', 'threads': 4},
             'reference_inclusive_candidate_ranks_are_recognition_accuracy': False,
             'token_length_normalized_scores_are_actual_log_probabilities': False}
    save_new(output, value)
    print(json.dumps(bind(output)), flush=True)


if __name__ == '__main__':
    main()
