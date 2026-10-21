"""Evaluate fixed train-only memorization milestones without optimization.

This evaluator preserves full FP32 frame log probabilities for an independent
CPU CTC audit. The teacher cache is validated as training provenance but never
fed to inference; only the student's pixel input is evaluated.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
from pathlib import Path
import time
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DECODER_PATH = ROOT / 'experiments/local_training_step05_20261007/evaluate_prediction_decode.py'
DECODER_SHA = '9645a8fc56edd45f2a6e7e7765f56f30b3cae8e8b5433709daee91a1903065d2'
if hashlib.sha256(DECODER_PATH.read_bytes()).hexdigest() != DECODER_SHA:
    raise ValueError('Frozen prediction decoder changed')
spec = importlib.util.spec_from_file_location('step11_frozen_tolerant_decoder', DECODER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(str(DECODER_PATH))
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)
original, runner, core = decoder.original, decoder.runner, decoder.core
require, bind, verify = runner.require, original.bind, original.verify
MILESTONES = (0, 300, 600, 900, 1200)
CONDITIONS = ('normal', 'static', 'shuffled')


def evaluation_code() -> list[dict]:
    """All executable evaluator and imported helper sources, before training."""
    paths = [Path(__file__), BASE / 'test_evaluate.py', DECODER_PATH,
             DECODER_PATH.parent / 'evaluate.py',
             ROOT / 'experiments/olkavs_official/diagnostic_utils.py',
             ROOT / 'experiments/olkavs_official/evaluation_metrics.py']
    paths.extend(Path(item['path']) for item in runner._code_bindings())
    return [bind(path) for path in dict.fromkeys(paths)]


def evaluation_contract() -> dict:
    """Return the exact diagnostic, which uses no development or test data."""
    return {'schema': 'lipnet-step11-train-evaluation-contract-v1',
            'milestones': list(MILESTONES), 'normal_at_all_milestones': True,
            'static_and_shuffled_at_update': 1200, 'records_per_run': 56,
            'groups': ['all8', 'old4', 'new4'],
            'primary_success': 'Final normal exact sentences equal 8 of 8; no early stop or checkpoint selection',
            'decoder': 'Argmax, collapse adjacent repeats before blank0 removal, compose_jamos(strict=False), retain U+FFFD',
            'primary_metric': 'Pooled NFC-strip CER including internal spaces; failed inference remains empty prediction',
            'secondary_metrics': ['WER', 'exact sentences', 'collapsed-token edit rate', 'CPU FP32 CTC mean loss',
                                  'output lengths', 'blank argmax frequency', 'mean blank posterior', 'repetition and histograms'],
            'ctc': 'Eval-mode FP32 log probabilities; CPU ctc_loss(blank=0,reduction=mean,zero_infinity=False), normalized by target-token length per clip',
            'posterior_evidence': 'All successful T-by-69 FP32 log probabilities saved in immutable compressed NPZ; no object arrays',
            'failure_policy': 'Artifact/identity/state violations abort; inference failure retains fixed denominator, no posterior or fabricated CTC',
            'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
            'limits': ['Train8 memorization is not unseen-speaker or webcam accuracy',
                       'Four new train labels remain provisionally selected, not human-approved',
                       'Static/shuffle probes cannot isolate lip motion from every temporal cue']}


def optimization_contract(arm: str) -> dict:
    require(arm in core.ARMS, 'Unknown arm')
    return {'optimizer': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8,
            'weight_decay': 0., 'gradient_clip': 1., 'accumulation_clips': 2,
            'lambda': 0. if arm == 'A_ctc' else 1., 'ctc_device': 'cpu', 'precision': 'float32'}


def validate_checkpoint_payload(plan, validation, arm, payload, expected_update, initial_hashes):
    """Require complete provenance and state for each fixed saved milestone."""
    require(expected_update in MILESTONES[1:], 'Unplanned checkpoint milestone')
    require(plan['updates'] == 1200 and plan['checkpoint_interval'] == 300 and plan['batch_size'] == 2,
            'Changed diagnostic budget')
    require(payload.get('schema') == runner.SCHEMA and payload['completed_updates'] == expected_update,
            'Checkpoint position/schema mismatch')
    require(validation['plan_sha256'] == runner.canonical_plan_hash(plan)
            and plan.get('plan_sha256') == validation['plan_sha256'], 'Plan digest mismatch')
    expected_identity = {'plan_sha256': validation['plan_sha256'], 'seed': plan['seed'], 'arm': arm,
                         'total_updates': plan['updates'], 'batch_size': 2,
                         'schedule_sha256': runner.nested_hash(plan['schedule']),
                         'data_sha256': validation['data_sha256'], 'clip_bindings': validation['clips'],
                         'code_bindings': runner._code_bindings(), 'runtime': plan['runtime'],
                         'optimization': optimization_contract(arm)}
    require(payload['identity'] == expected_identity, 'Checkpoint identity differs from plan/data/code/arm/runtime')
    require(payload['initial_state_hashes'] == initial_hashes, 'Wrong fresh initialization')
    history = payload['history']
    require(len(history) == expected_update and all(row['update'] == i + 1 and row['ids'] == plan['schedule'][i]
            for i, row in enumerate(history)), 'Checkpoint history/schedule mismatch')
    actual = runner.state_hashes(payload['state'])
    require(actual == payload['state_hashes'] == history[-1]['state_hashes'], 'Checkpoint full state mismatch')
    return expected_identity


def validate_groups(plan, clips):
    groups = plan['diagnostic_groups']
    require(set(groups) == {'old4', 'new4'} and all(len(value) == len(set(value)) == 4 for value in groups.values()),
            'Expected old4 and new4 train groups')
    require(set(groups['old4']).isdisjoint(groups['new4']) and set(clips) == set(groups['old4'] + groups['new4']),
            'Changed diagnostic group membership')
    for sid, clip in clips.items():
        require(clip['id'] == sid and clip['split'] == 'train' and clip.get('source_split', 'train') == 'train',
                'Non-training diagnostic input')
        require('audio' not in clip and clip['tokens'] == decoder.tokenizer.encode_text(clip['text']),
                'Audio input or transcript-token mismatch')
    return groups


def run_lengths(text):
    """Count same nonspace characters; whitespace always separates runs."""
    lengths, previous, length = [], None, 0
    for char in text:
        if not char.isspace() and char == previous:
            length += 1
        else:
            if length:
                lengths.append(length)
            length = 0 if char.isspace() else 1
        previous = char
    if length:
        lengths.append(length)
    return lengths


def output_diagnostics(lp, decoded, reference):
    require(isinstance(lp, np.ndarray) and lp.dtype == np.float32 and lp.ndim == 2
            and lp.shape[1] == 69 and len(lp) > 0 and np.isfinite(lp).all(), 'Invalid FP32 log probabilities')
    # The model already returns log_softmax; do not silently renormalize it.
    lp_tensor = torch.from_numpy(lp)
    normalization_error = float(torch.logsumexp(lp_tensor, dim=-1).abs().max())
    require(normalization_error <= 1e-4 and bool((lp_tensor <= 1e-5).all()), 'Model output is not normalized log probabilities')
    blank = lp_tensor[:, 0].exp().tolist()
    raw, collapsed = decoded['raw_argmax_ids'], decoded['collapsed_ids']
    prediction = unicodedata.normalize('NFC', decoded['prediction']).strip()
    reference = unicodedata.normalize('NFC', reference).strip()
    lengths = run_lengths(prediction)
    histogram = lambda values: {str(key): value for key, value in sorted(Counter(values).items())}
    return {'frames': len(lp), 'blank_argmax_frames': raw.count(0), 'blank_argmax_fraction': raw.count(0) / len(lp),
            'blank_posterior_per_frame': blank, 'mean_blank_posterior': float(np.mean(blank)),
            'max_abs_logsumexp_error': normalization_error,
            'raw_token_histogram': histogram(raw), 'collapsed_token_histogram': histogram(collapsed),
            'character_histogram': histogram(prediction),
            'nonspace_character_histogram': histogram(char for char in prediction if not char.isspace()),
            'prediction_characters': len(prediction), 'reference_characters': len(reference),
            'prediction_nonspace_characters': sum(not char.isspace() for char in prediction),
            'reference_nonspace_characters': sum(not char.isspace() for char in reference),
            'longest_nonspace_same_character_run': max(lengths, default=0),
            'same_character_runs_ge5': sum(length >= 5 for length in lengths)}


def summarize_records(records):
    """Pool integer errors; do not treat a missing inference loss as zero."""
    successful = [row for row in records if row['status'] == 'ok']
    diagnostic = [row['output_diagnostics'] for row in successful]
    frames = sum(row['frames'] for row in diagnostic)
    blank_frames = sum(row['blank_argmax_frames'] for row in diagnostic)
    blank_sum = sum(sum(row['blank_posterior_per_frame']) for row in diagnostic)
    target_tokens = sum(row['token_metrics']['target_tokens'] for row in records)
    token_edits = sum(row['token_metrics']['token_edits'] for row in records)
    histograms = {}
    for key in ('raw_token_histogram', 'collapsed_token_histogram', 'character_histogram', 'nonspace_character_histogram'):
        total = Counter()
        for row in diagnostic:
            total.update(row[key])
        histograms[key] = dict(sorted(total.items()))
    return {'aggregate': core.aggregate(records), 'diagnostics': {
        'successful_inferences': len(successful), 'failed_inferences': len(records) - len(successful),
        'ctc_loss_count': len(successful),
        'ctc_loss_mean_over_successful_clips': float(np.mean([row['ctc_loss_mean'] for row in successful])) if successful else None,
        'token_edits': token_edits, 'target_tokens': target_tokens,
        'token_error_percent': 100 * token_edits / target_tokens,
        'token_exact_sentences': sum(row['token_metrics']['exact'] for row in records),
        'frames': frames, 'blank_argmax_frames': blank_frames,
        'blank_argmax_fraction': blank_frames / frames if frames else None,
        'blank_posterior_sum': blank_sum, 'mean_blank_posterior': blank_sum / frames if frames else None,
        'prediction_characters': sum(row['prediction_characters'] for row in diagnostic),
        'reference_characters': sum(row['metrics']['reference_characters'] for row in records),
        'replacement_characters': sum(row['replacement_characters'] for row in records),
        'sentences_with_same_character_run_ge5': sum(row['same_character_runs_ge5'] > 0 for row in diagnostic),
        'longest_nonspace_same_character_run': max((row['longest_nonspace_same_character_run'] for row in diagnostic), default=0),
        **histograms}}


def evaluate_clips(model, clips, groups, conditions, device, update, posteriors):
    """Evaluate every fixed train clip and retain full denominators on failure."""
    ids = groups['old4'] + groups['new4']
    require(len(ids) == len(set(ids)) == 8 and set(ids) == set(clips), 'Changed evaluation cohort')
    require(conditions in (('normal',), CONDITIONS), 'Unplanned conditions')
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'Student must be frozen eval')
    require(all(clips[sid]['split'] == 'train' for sid in ids), 'Nontraining evaluation input')
    before = runner.nested_hash(model.state_dict())
    values = {}
    with torch.inference_mode():
        for condition in conditions:
            records = []
            for position, sid in enumerate(ids):
                clip = clips[sid]
                started = time.perf_counter()
                decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                ctc, diagnostic, error, temporal, posterior_key = None, None, None, None, None
                try:
                    pixels = clip['pixels']
                    temporal = original.condition_metadata(pixels, condition, sid)
                    lp = model(core.prepare_student(core.perturb_pixels(pixels, condition, sid), device))
                    require(lp.shape == (len(pixels), 1, 69) and torch.isfinite(lp).all().item(), 'Malformed/nonfinite model output')
                    lp_cpu = lp.float().cpu()
                    lp_array = lp_cpu[:, 0].numpy().copy()
                    candidate = decoder.decode_prediction(lp_array)
                    diagnostic = output_diagnostics(lp_array, candidate, clip['text'])
                    ctc = float(F.ctc_loss(lp_cpu, torch.tensor(clip['tokens'], dtype=torch.long),
                                torch.tensor([len(pixels)]), torch.tensor([len(clip['tokens'])]),
                                blank=0, reduction='mean', zero_infinity=False))
                    require(np.isfinite(ctc) and ctc >= -1e-6, 'Nonfinite/negative CTC evaluation loss')
                    decoded = candidate
                    posterior_key = f'u{update:04d}_{condition}_{position}'
                    require(posterior_key not in posteriors, 'Duplicate posterior archive key')
                    posteriors[posterior_key] = lp_array
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                    ctc, diagnostic, posterior_key = None, None, None
                token_edits = decoder.diagnostics.edit_distance(clip['tokens'], decoded['collapsed_ids'])
                records.append({'id': sid, 'speaker_id': clip['speaker_id'], 'source_stem': clip['source_stem'],
                    'split': 'train', 'source_split': clip.get('source_split', 'train'),
                    'group': 'old4' if sid in groups['old4'] else 'new4', 'condition': condition,
                    'reference': clip['text'], 'prediction': decoded['prediction'],
                    'status': 'ok' if error is None else 'failed', 'error': error, 'frames': len(clip['pixels']),
                    'temporal': temporal, 'seconds': time.perf_counter() - started,
                    'raw_argmax_ids': decoded['raw_argmax_ids'], 'collapsed_ids': decoded['collapsed_ids'],
                    'target_token_ids': clip['tokens'], 'replacement_characters': decoded['replacement_characters'],
                    'strict_label_decoder_diagnostic': decoded['strict_replay'],
                    'metrics': core.counts(clip['text'], decoded['prediction']), 'ctc_loss_mean': ctc,
                    'token_metrics': {'token_edits': token_edits, 'target_tokens': len(clip['tokens']),
                                      'exact': clip['tokens'] == decoded['collapsed_ids']},
                    'output_diagnostics': diagnostic, 'log_posteriors_key': posterior_key})
            values[condition] = {'records': records, 'groups': {'all8': summarize_records(records),
                **{group: summarize_records([row for row in records if row['id'] in selected]) for group, selected in groups.items()}}}
    after = runner.nested_hash(model.state_dict())
    require(before == after, 'Evaluation changed student state')
    return {'completed_updates': update, 'conditions': values, 'model_state_before': before,
            'model_state_after': after, 'model_state_unchanged': True}


def validate_checkpoint_refs(checkpoints):
    require([item['completed_updates'] for item in checkpoints] == list(MILESTONES[1:]),
            'Missing, duplicate, reordered or extra checkpoint milestone')
    for item in checkpoints:
        path = Path(item['path']).resolve()
        require(Path(item['sidecar']).resolve() == path.with_suffix('.pt.json'), 'Wrong checkpoint sidecar path')
        require(bind(path)['sha256'] == item['sha256'] and bind(item['sidecar'])['sha256'] == item['sidecar_sha256'],
                'Checkpoint reference SHA mismatch')


def evaluate_run(plan, clips, arm, checkpoint_refs, initial_state_hashes_expected, output,
                 device='cuda:0', experiment_plan_ref=None):
    """Verify and evaluate a completed run; no checkpoint ranking or selection."""
    output = Path(output).resolve()
    archive_path = output.with_suffix('.posteriors.npz')
    require(output.is_relative_to(BASE / 'results') and not output.exists() and not archive_path.exists(),
            'Use a new immutable Step11 results path')
    require(plan['updates'] == 1200 and plan['checkpoint_interval'] == 300 and plan['seed'] in (42, 43), 'Wrong diagnostic recipe')
    validation = runner.validate_plan(plan, clips, arm)
    require(plan['training_data'] == {key: validation[key] for key in ('data_sha256', 'clips')}, 'Changed train input bytes')
    groups = validate_groups(plan, clips)
    require(plan['initial_state_hashes_expected'][arm] == initial_state_hashes_expected, 'Unbound initial state expectation')
    frozen = {ref['path']: ref for ref in plan['execution_code']}
    for ref in evaluation_code():
        require(frozen.get(ref['path']) == ref, 'Evaluator code was not frozen before training')
    for ref in plan['execution_code']:
        verify(ref)
    if experiment_plan_ref is not None:
        verify(experiment_plan_ref)
        require(plan['experiment_plan'] == experiment_plan_ref, 'Wrong experiment policy reference')
    validate_checkpoint_refs(checkpoint_refs)
    core.setup_runtime()
    selected_device = torch.device(device)
    runtime = runner._runtime(selected_device)
    require(runtime == plan['runtime'], 'Evaluation runtime changed')
    outer_rng = core.capture_rng()
    began = time.perf_counter()
    posterior_arrays, results = {}, []
    try:
        model, adapter, optimizer = core.make_objects(plan['seed'], arm, selected_device)
        actual_initial = runner.state_hashes(core.runtime_state(model, adapter, optimizer))
        require(actual_initial == initial_state_hashes_expected, 'Reconstructed initialization differs')
        model.eval().requires_grad_(False)
        initial = evaluate_clips(model, clips, groups, ('normal',), selected_device, 0, posterior_arrays)
        initial['checkpoint'] = None
        results.append(initial)
        del adapter, optimizer
        for checkpoint in checkpoint_refs:
            update = checkpoint['completed_updates']
            payload = runner.load_checkpoint(Path(checkpoint['path']))
            validate_checkpoint_payload(plan, validation, arm, payload, update, initial_state_hashes_expected)
            require(checkpoint['state_hashes'] == payload['state_hashes'], 'Checkpoint reference state mismatch')
            model.load_state_dict(payload['state']['model'], strict=True)
            require(runner.nested_hash(model.state_dict()) == payload['state_hashes']['model'], 'Loaded student state mismatch')
            value = evaluate_clips(model, clips, groups, CONDITIONS if update == 1200 else ('normal',),
                                   selected_device, update, posterior_arrays)
            value['checkpoint'] = checkpoint
            results.append(value)
            del payload
        if selected_device.type == 'cuda':
            torch.cuda.synchronize(selected_device)
    finally:
        core.restore_rng(outer_rng)
    require(runner.nested_hash(core.capture_rng()) == runner.nested_hash(outer_rng), 'Evaluation changed caller RNG')
    row_count = sum(len(condition['records']) for item in results for condition in item['conditions'].values())
    require(row_count == 56, 'Incomplete milestone evaluation')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **posterior_arrays)
    runner._atomic_bytes(archive_path, buffer.getvalue())
    value = {'schema': 'lipnet-step11-train-memorization-evaluation-v1', 'status': 'completed',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': plan['seed'], 'arm': arm,
             'plan_sha256': validation['plan_sha256'], 'data_sha256': validation['data_sha256'],
             'experiment_plan': experiment_plan_ref, 'initial_state_hashes_verified': actual_initial,
             'ids': groups['old4'] + groups['new4'], 'diagnostic_groups': groups, 'milestones': results,
             'record_count': row_count, 'log_posteriors_archive': bind(archive_path),
             'posterior_arrays': len(posterior_arrays), 'runtime': runtime, 'evaluation_code': evaluation_code(),
             'evaluation_contract': evaluation_contract(), 'elapsed_seconds': time.perf_counter() - began,
             'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
             'dev_inputs_loaded': False, 'held_inputs_loaded': False, 'test_inputs_loaded': False,
             'service_weights_updated': False, 'caller_rng_restored_exactly': True}
    runner._save_json(output, value)
    return bind(output)
