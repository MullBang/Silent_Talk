"""Stage14: restore the same step6400 state and compare two fixed learning rates.

Both branches continue the identical schedule through9600. Only LR is changed;
the source Adam moments, scaler and all RNG states are restored, not restarted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'scripts')]
import diagnose_learning_transition as prior
import diagnose_learning_budget as budget
from models.dataset_manifest import load_manifest, sha256_file
from transition_diagnostics import make_training_schedule, parameter_groups, select_transition_cohorts
from learning_budget_state import (save_training_state, load_training_state, capture_rng_state,
    _backend_state, compare_nested, compare_state_dicts)

PRIOR_RUN = ROOT / 'results/step13-budget/run01'
PRIOR_SUMMARY = ROOT / 'docs/단계별_개선/13_검증결과.json'
START, END = 6400, 9600
BRANCHES = {'A_keep': .0003, 'B_reduce': .0001}
STEPS = [6400, 6401, 6500, 6800, 7200, 8000, 8800, 9600]
CONTROL_STEPS = [6400, 7200, 8000, 9600]
BEAM_STEPS = [6400, 8000, 9600]
STATE_STEPS = [6400, 8000, 9600]
SOURCE_PATHS = [*budget.SOURCE_PATHS, 'scripts/diagnose_learning_rate.py']
RUNTIME_FIELDS = ('model_state_dict', 'optimizer_state_dict', 'scaler_state_dict', 'rng_state',
                  'model_training', 'module_training', 'backend_state')
require, write_json, read_json = prior.require, prior.write_json, prior.read_json


def source_hashes() -> dict:
    return {name: sha256_file(ROOT / name) for name in SOURCE_PATHS}


def runtime_state(model, optimizer, scaler) -> dict:
    """Snapshot comparison views without random draws, forwards or state changes."""
    return {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(), 'rng_state': capture_rng_state(),
        'model_training': model.training,
        'module_training': {name: module.training for name, module in model.named_modules()},
        'backend_state': _backend_state()}


def restore_branch(path, *, model, optimizer, scaler, expected_contract: dict, learning_rate: float) -> dict:
    """Require exact full-state restoration before making the one allowed change."""
    require(learning_rate in BRANCHES.values(), 'Unplanned learning rate')
    source = load_training_state(path, model=model, optimizer=optimizer, scaler=scaler,
                                 expected_contract=expected_contract)
    expected = {key: source[key] for key in RUNTIME_FIELDS}
    before = compare_nested(expected, runtime_state(model, optimizer, scaler))
    require(before['exact'], 'Source runtime did not restore exactly')
    before_lrs = [group['lr'] for group in optimizer.param_groups]
    expected = copy.deepcopy(expected)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    for group in expected['optimizer_state_dict']['param_groups']:
        group['lr'] = learning_rate
    after = compare_nested(expected, runtime_state(model, optimizer, scaler))
    require(after['exact'], 'Fork changed runtime beyond learning rate')
    return {'status': 'passed', 'source_runtime_exact': before,
            'lr_only_runtime_exact': after, 'learning_rate': learning_rate, 'before_lrs': before_lrs}


def schedule_metadata(schedule: dict) -> dict:
    return {k: schedule[k] for k in ('sample_ids', 'attempt_updates', 'batch_size', 'seed', 'batches_sha256')}


def source_contract(previous: dict, completed: dict) -> dict:
    return {'attempt': START, 'source_hashes': previous['source_hashes'], 'settings': previous['settings'],
        'schedule': schedule_metadata(previous['schedule']),
        'provenance': {'mode': 'smoke_only', 'purpose': 'train_subset_budget_diagnostic',
            'manifest_sha256': previous['manifest_sha256'], 'sample_ids': previous['cohorts']['expanded64'],
            'generalization_evaluation': False, 'deployment': False, 'plan_sha256': completed['plan_sha256']},
        'counters': {k: completed[k] for k in ('optimizer_steps', 'skipped_optimizer_steps',
                    'attempted_exposures', 'successful_update_exposures')}}


def create_plan(out: Path) -> dict:
    require(not out.exists(), 'Output exists; use a new output directory')
    previous, completed = read_json(PRIOR_RUN / 'plan.json'), read_json(PRIOR_RUN / 'completed.json')
    verified = read_json(PRIOR_SUMMARY)
    quality = read_json(PRIOR_RUN / 'quality_checks.json')
    require(verified['status'] == 'completed_and_verified' and completed['status'] == 'completed', 'Prior incomplete')
    require(sha256_file(PRIOR_SUMMARY) == quality['artifact_sha256'][PRIOR_SUMMARY.relative_to(ROOT).as_posix()],
            'Prior verified summary changed')
    require(sha256_file(PRIOR_RUN / 'completed.json') == verified['provenance']['completed_sha256'], 'Prior completion changed')
    require(sha256_file(PRIOR_RUN / 'plan.json') == completed['plan_sha256'], 'Prior plan changed')
    require(budget.source_hashes() == previous['source_hashes'], 'Stage13 frozen sources changed')
    environment = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'numpy': np.__version__, 'python': sys.version}
    require(environment == completed['environment'], 'Prior execution environment changed')
    document, metadata = load_manifest(prior.MANIFEST, verify_files=False)
    require(metadata['manifest_sha256'] == previous['manifest_sha256'] == prior.MANIFEST_SHA, 'Manifest changed')
    require(select_transition_cohorts(document['records']) == previous['cohorts'], 'Train64 changed')
    ids = previous['cohorts']['expanded64']
    by_id = {row['id']: row for row in document['records']}
    require({s: by_id[s] for s in ids} == {r['id']: r for r in previous['records']}, 'Train records changed')
    files = ['plan.json', 'completed.json', 'prefix_gate.json', 'telemetry.jsonl']
    require(sha256_file(PRIOR_RUN / 'prefix_gate.json') == completed['prefix_gate_sha256'], 'Prior gate changed')
    require(sha256_file(PRIOR_RUN / 'telemetry.jsonl') == completed['telemetry_sha256'], 'Prior telemetry changed')
    for spec in completed['snapshots']:
        folder = PRIOR_RUN / spec['folder']
        summary = read_json(folder / 'summary.json')
        require(sha256_file(folder / 'summary.json') == spec['summary_sha256'], 'Prior snapshot changed')
        for filename, field in [('rows.jsonl', 'rows_sha256'), ('posteriors.npz', 'posteriors_sha256')]:
            require(sha256_file(folder / filename) == summary[field], 'Prior outputs changed')
        files.extend((folder / name).relative_to(PRIOR_RUN).as_posix() for name in ('summary.json', 'rows.jsonl', 'posteriors.npz'))
        checkpoint = Path(spec['checkpoint']) if spec['step'] == 0 else PRIOR_RUN / spec['checkpoint']
        require(sha256_file(checkpoint) == spec['checkpoint_sha256'], 'Prior checkpoint changed')
        if spec['step']:
            files.append(spec['checkpoint'])
    for spec in completed['resume_states']:
        require(sha256_file(PRIOR_RUN / spec['file']) == spec['sha256'], 'Prior full state changed')
        files.append(spec['file'])
    path = PRIOR_RUN / 'training_state_6400.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    contract = source_contract(previous, completed)
    require(compare_nested(contract, {k: state[k] for k in contract})['exact'], 'Source full-state contract mismatch')
    require(compare_state_dicts(torch.load(PRIOR_RUN / 'step_6400.pt', map_location='cpu', weights_only=True)['state_dict'],
                               state['model_state_dict'])['exact'], 'Source checkpoint disagrees with full state')
    schedule = make_training_schedule(ids, attempt_updates=END)
    require(schedule['batches'][:START] == previous['schedule']['batches'], '6400 schedule prefix changed')
    settings = {**previous['settings'], 'attempt_updates': END, 'fresh_optimizer_per_branch': False,
                'grad_scaler_init_scale': state['scaler_state_dict']['scale'],
                'normal_steps': STEPS, 'control_steps': CONTROL_STEPS, 'beam_steps': BEAM_STEPS}
    plan = {'schema': 'lipnet-learning-rate-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'prior_run': str(PRIOR_RUN), 'prior_summary': str(PRIOR_SUMMARY), 'prior_summary_sha256': sha256_file(PRIOR_SUMMARY),
        'prior_files': {name: sha256_file(PRIOR_RUN / name) for name in files},
        'prior_plan_sha256': completed['plan_sha256'], 'prior_state': str(path), 'prior_state_sha256': sha256_file(path),
        'source_contract': contract, 'source_hashes': source_hashes(), 'environment': environment,
        **{key: previous[key] for key in ('manifest', 'manifest_sha256', 'cohorts', 'records', 'scope', 'nonfinite_policy')},
        'settings': settings, 'branches': {key: {'learning_rate': lr, 'settings': {**settings, 'learning_rate': lr}}
                                          for key, lr in BRANCHES.items()},
        'schedule': schedule, 'start_attempt': START, 'continuation_start': START+1,
        'additional_attempt_updates': END-START, 'resume_state_steps': STATE_STEPS,
        'initial_counters': state['counters'], 'initial_scaler': state['scaler_state_dict'],
        'preflight_policy': 'Both branches restore source exactly, change only LR, save full initial state and reproduce all step6400 outputs before either branch trains.',
        'interpretation': ['Two learning-rate continuations of one shared training state, not independent random seeds.',
            'Same planned batch schedule and budget; LR can cause different later AMP skips and successful exposures.',
            'Fixed endpoints; no early stopping, best checkpoint selection, validation/test cache or deployment.']}
    out.mkdir(parents=True)
    write_json(out / 'plan.json', plan)
    return plan


@torch.inference_mode()
def evaluate_snapshot(model: torch.nn.Module, plan: dict, cache: dict, records: dict, step: int,
                      folder: Path, checkpoint_sha: str) -> dict:
    """Evaluate fixed stage14 snapshots with the unchanged stage13 math."""
    folder.mkdir()
    started = time.perf_counter()
    conditions = ["normal", "static_middle", "shuffle"] if step in CONTROL_STEPS else ["normal"]
    beam_performed = step in BEAM_STEPS
    arrays, rows = {}, []
    with prior.preserve_rng_and_mode(model):
        for index, sample_id in enumerate(plan["cohorts"]["expanded64"]):
            source, item = records[sample_id], cache[sample_id]
            normal, normal_text = None, None
            for condition in conditions:
                frames, transform = prior.transform_frames(item["frames"], condition, sample_id=sample_id, seed=42)
                lp, _ = prior.forward(model, frames, capture=False)
                greedy = prior.ctc_greedy_decode(lp)
                beam = prior.ctc_beam_search_decode(lp, beam_width=10) if beam_performed else None
                diagnostics = prior.analyze_ctc(lp, item["label"], beam_text=beam if beam_performed else "")
                if not beam_performed:
                    diagnostics = {k: v for k, v in diagnostics.items() if not k.startswith("beam")}
                target = torch.tensor(item["label"], dtype=torch.long)
                loss = torch.nn.functional.ctc_loss(torch.from_numpy(lp[:, None, :]), target,
                    torch.tensor([len(lp)]), torch.tensor([len(target)]), blank=0, reduction="mean", zero_infinity=False)
                require(bool(torch.isfinite(loss)), "Nonfinite evaluation loss")
                name = "shuffle_42" if condition == "shuffle" else condition
                key = f"sample_{index:03d}_{name}"
                arrays[key] = lp
                row = {"id": sample_id, "step": step, "split": "train", "condition": name,
                    "cohort": "short16" if sample_id in plan["cohorts"]["short16"] else "additional48",
                    "speaker_id": source["speaker_id"], "reference": source["text"], "frames": len(lp),
                    "greedy": prior.prediction_row(source, greedy),
                    "beam": prior.prediction_row(source, beam) if beam_performed else None,
                    "ctc": diagnostics, "ctc_loss_per_target_token": loss.item(),
                    "posterior_key": key, "transform": transform, "comparison_to_normal": None}
                if condition == "normal":
                    normal, normal_text = lp, greedy
                else:
                    row["comparison_to_normal"] = prior.compare_posteriors(normal, lp,
                        normal_text=normal_text, control_text=greedy)
                rows.append(row)
    with (folder / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    summary = {"step": step, "beam_performed": beam_performed,
        "conditions": ["shuffle_42" if c == "shuffle" else c for c in conditions],
        "rows": len(rows), "checkpoint_sha256": checkpoint_sha,
        "rows_sha256": sha256_file(folder / "rows.jsonl"),
        "posteriors_sha256": sha256_file(folder / "posteriors.npz"),
        "seconds": time.perf_counter() - started, "cohorts": {}}
    for cohort in ("short16", "additional48"):
        normal_rows = [r for r in rows if r["cohort"] == cohort and r["condition"] == "normal"]
        summary["cohorts"][cohort] = {"count": len(normal_rows),
            "greedy_exact": sum(r["greedy"]["exact"] for r in normal_rows),
            "greedy_mean_cer": float(np.mean([r["greedy"]["cer"] for r in normal_rows]))}
    write_json(folder / "summary.json", summary)
    print(json.dumps({"evaluation_step": step, **summary["cohorts"], "seconds": summary["seconds"]}), flush=True)
    return summary


def compare_initial_outputs(folder: Path) -> dict:
    old = PRIOR_RUN / 'step_6400'
    read_rows = lambda p: [json.loads(x) for x in (p / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    rows_exact = read_rows(old) == read_rows(folder)
    arrays_exact, count, delta = True, 0, 0.
    with np.load(old / 'posteriors.npz', allow_pickle=False) as a, np.load(folder / 'posteriors.npz', allow_pickle=False) as b:
        arrays_exact = a.files == b.files
        if arrays_exact:
            for key in a.files:
                same = a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and np.array_equal(a[key], b[key])
                arrays_exact = arrays_exact and same
                if a[key].shape == b[key].shape:
                    delta = max(delta, float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64)))))
                count += 1
    return {'prediction_rows_exact': rows_exact, 'posterior_arrays_exact': arrays_exact,
            'posterior_arrays_compared': count, 'posterior_max_absolute_difference': delta}


def provenance(plan: dict, key: str) -> dict:
    return {'mode': 'smoke_only', 'purpose': 'train_subset_lr_diagnostic', 'branch': key,
        'manifest_sha256': plan['manifest_sha256'], 'sample_ids': plan['cohorts']['expanded64'],
        'generalization_evaluation': False, 'deployment': False}


def save_snapshot(out, model, plan, plan_sha, key, step, counters, cache, records):
    checkpoint = out / f'step_{step:04d}.pt'
    model.save_checkpoint(str(checkpoint), training_data=provenance(plan, key), training_run={
        'lr_diagnostic': True, 'branch': key, 'attempt_updates': step, 'additional_attempt_updates': step-START,
        'optimizer_steps': counters['optimizer_steps'], 'skipped_optimizer_steps': counters['skipped_optimizer_steps'],
        'plan_sha256': plan_sha, 'source_training_state_sha256': plan['prior_state_sha256'],
        'inference_only': True, 'selection': 'fixed_attempt_schedule', 'settings': plan['branches'][key]['settings']})
    digest = sha256_file(checkpoint)
    folder = out / f'step_{step:04d}'
    evaluate_snapshot(model, plan, cache, records, step, folder, digest)
    return {'step': step, 'folder': folder.name, 'checkpoint': checkpoint.name,
            'checkpoint_sha256': digest, 'summary_sha256': sha256_file(folder / 'summary.json')}


def save_state(out, model, optimizer, scaler, plan, plan_sha, key, step, counters):
    path = out / f'training_state_{step:04d}.pt'
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, attempt=step,
        schedule=schedule_metadata(plan['schedule']), provenance={**provenance(plan, key), 'plan_sha256': plan_sha},
        source_hashes=plan['source_hashes'], settings=plan['branches'][key]['settings'], counters=counters)
    return {'step': step, 'file': path.name, 'sha256': sha256_file(path)}


def new_objects():
    model = prior.LipNetBaseline().to('cuda').train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024)
    return model, optimizer, scaler


def preflight(out, plan, cache, records):
    plan_sha = sha256_file(out / 'plan.json')
    gates, snapshots, states = {}, {}, {}
    for key, lr in BRANCHES.items():
        folder = out / key
        folder.mkdir()
        model, optimizer, scaler = new_objects()
        gate = restore_branch(plan['prior_state'], model=model, optimizer=optimizer, scaler=scaler,
            expected_contract=plan['source_contract'], learning_rate=lr)
        states[key] = save_state(folder, model, optimizer, scaler, plan, plan_sha, key, START, plan['initial_counters'])
        snapshots[key] = save_snapshot(folder, model, plan, plan_sha, key, START, plan['initial_counters'], cache, records)
        saved = torch.load(folder / states[key]['file'], map_location='cpu', weights_only=True)
        gate['evaluation_preserved_runtime'] = compare_nested({k: saved[k] for k in RUNTIME_FIELDS}, runtime_state(model, optimizer, scaler))
        gate['initial_outputs'] = compare_initial_outputs(folder / 'step_6400')
        gate['source_training_state_sha256'] = plan['prior_state_sha256']
        gate['initial_training_state_sha256'] = states[key]['sha256']
        write_json(folder / 'restore_gate.json', gate)
        require(gate['evaluation_preserved_runtime']['exact'] and gate['initial_outputs']['prediction_rows_exact']
                and gate['initial_outputs']['posterior_arrays_exact'], 'Initial output/runtime reproduction failed')
        gates[key] = gate
        del model, optimizer, scaler, saved
        torch.cuda.empty_cache()
    a = torch.load(out / 'A_keep/training_state_6400.pt', map_location='cpu', weights_only=True)
    b = torch.load(out / 'B_reduce/training_state_6400.pt', map_location='cpu', weights_only=True)
    expected = {k: a[k] for k in RUNTIME_FIELDS}
    for group in expected['optimizer_state_dict']['param_groups']:
        group['lr'] = BRANCHES['B_reduce']
    cross = compare_nested(expected, {k: b[k] for k in RUNTIME_FIELDS})
    require(cross['exact'], 'Branches differ initially beyond learning rate')
    write_json(out / 'preflight_gate.json', {'status': 'passed', 'plan_sha256': plan_sha,
        'checked_before_either_branch_training': True, 'cross_branch_runtime_lr_only_exact': cross,
        'restore_gate_sha256': {key: sha256_file(out / key / 'restore_gate.json') for key in BRANCHES},
        'initial_states': states, 'initial_snapshots': snapshots})
    print('PREFLIGHT PASSED: both source restores, full runtime preservation and all initial outputs exact', flush=True)
    return states, snapshots


def run_branch(out, plan, key, initial_state, initial_snapshot, cache, records):
    folder = out / key
    plan_sha = sha256_file(out / 'plan.json')
    path = folder / initial_state['file']
    require(sha256_file(path) == initial_state['sha256'], 'Prepared initial state changed')
    model, optimizer, scaler = new_objects()
    contract = {'attempt': START, 'source_hashes': plan['source_hashes'],
        'settings': plan['branches'][key]['settings'], 'schedule': schedule_metadata(plan['schedule']),
        'provenance': {**provenance(plan, key), 'plan_sha256': plan_sha}, 'counters': plan['initial_counters']}
    gate = restore_branch(path, model=model, optimizer=optimizer, scaler=scaler,
                          expected_contract=contract, learning_rate=BRANCHES[key])
    write_json(folder / 'start_gate.json', {**gate, 'initial_training_state_sha256': initial_state['sha256'],
        'preflight_gate_sha256': sha256_file(out / 'preflight_gate.json'), 'next_attempt': START+1})
    groups = parameter_groups(model)
    counters = copy.deepcopy(plan['initial_counters'])
    exposures = Counter(counters['attempted_exposures'])
    successful = Counter(counters['successful_update_exposures'])
    snapshots, states = [initial_snapshot], [initial_state]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (folder / 'telemetry.jsonl').open('x', encoding='utf-8') as stream:
        for entry in plan['schedule']['batches'][START:]:
            result = prior.optimizer_attempt(model, optimizer, scaler,
                prior.collate([cache[s] for s in entry['ids']]), groups, torch.device('cuda'))
            counters['optimizer_steps'] += int(not result['skipped'])
            counters['skipped_optimizer_steps'] += int(result['skipped'])
            exposures.update(entry['ids'])
            if not result['skipped']:
                successful.update(entry['ids'])
            counters.update(attempted_exposures=dict(exposures), successful_update_exposures=dict(successful))
            result.update(entry)
            result.update(optimizer_steps=counters['optimizer_steps'], skipped_optimizer_steps=counters['skipped_optimizer_steps'],
                frames=[records[s]['frames'] for s in entry['ids']], target_lengths=[len(cache[s]['label']) for s in entry['ids']])
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            step = entry['attempt']
            if step in STEPS:
                snapshots.append(save_snapshot(folder, model, plan, plan_sha, key, step, counters, cache, records))
            if step in STATE_STEPS:
                states.append(save_state(folder, model, optimizer, scaler, plan, plan_sha, key, step, counters))
            if step % 400 == 0:
                print(json.dumps({'branch': key, 'attempt': step, 'additional_attempts': step-START,
                    'optimizer_steps': counters['optimizer_steps'], 'skips': counters['skipped_optimizer_steps'],
                    'loss': result['loss'], 'seconds': time.perf_counter()-started}), flush=True)
    require(dict(exposures) == plan['schedule']['exposures'], 'Final exposure mismatch')
    completed = {'status': 'completed', 'branch': key, 'learning_rate': BRANCHES[key], 'plan_sha256': plan_sha,
        'attempt_updates': END, 'additional_attempt_updates': END-START, **counters,
        'additional_optimizer_steps': counters['optimizer_steps']-plan['initial_counters']['optimizer_steps'],
        'additional_skipped_optimizer_steps': counters['skipped_optimizer_steps']-plan['initial_counters']['skipped_optimizer_steps'],
        'snapshots': snapshots, 'resume_states': states,
        'telemetry_sha256': sha256_file(folder / 'telemetry.jsonl'),
        'restore_gate_sha256': sha256_file(folder / 'restore_gate.json'), 'start_gate_sha256': sha256_file(folder / 'start_gate.json'),
        'seconds': time.perf_counter()-started, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20, **plan['scope']}
    write_json(folder / 'completed.json', completed)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return completed


def run(out: Path):
    require(torch.cuda.is_available(), 'CUDA required')
    torch.set_num_threads(2)
    plan = create_plan(out)
    started = time.perf_counter()
    records = {r['id']: r for r in plan['records']}
    cache = {s: prior.load_train_cache(records[s]) for s in plan['cohorts']['expanded64']}
    states, snapshots = preflight(out, plan, cache, records)
    for key in BRANCHES:
        run_branch(out, plan, key, states[key], snapshots[key], cache, records)
    require(source_hashes() == plan['source_hashes'], 'Frozen source changed during experiment')
    for name, digest in plan['prior_files'].items():
        require(sha256_file(PRIOR_RUN / name) == digest, 'Stage13 artifact changed')
    require(sha256_file(PRIOR_SUMMARY) == plan['prior_summary_sha256']
            and sha256_file(prior.MANIFEST) == plan['manifest_sha256'], 'Input changed')
    write_json(out / 'completed.json', {'status': 'completed', 'plan_sha256': sha256_file(out / 'plan.json'),
        'preflight_gate_sha256': sha256_file(out / 'preflight_gate.json'),
        'branch_completed_sha256': {key: sha256_file(out / key / 'completed.json') for key in BRANCHES},
        'train_cache_ids_opened': list(cache), 'seconds': time.perf_counter()-started,
        'environment': plan['environment'], **plan['scope']})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/step14-learning-rate/run01')
    args = parser.parse_args()
    try:
        run(args.out.resolve())
    except Exception as exc:
        print(f'STAGE14 STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
