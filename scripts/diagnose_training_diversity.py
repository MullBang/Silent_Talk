"""Stage16: same restored optimizer state, fixed budget, training64 vs256.

Schedules and counters are local to this stage (0..3200). Source history is
bound separately at cumulative attempt9600. Adam/scaler/RNG are never reset.
The newly initialized *sampler* uses a local Random(42) in both branches.
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
import diagnose_learning_rate as lr
from diagnose_temporal_use import verify_cache, length_bin
from diversity_selection import select_diversity
from generalization_selection import normalize_transcript
from learning_budget_state import save_training_state, load_training_state, compare_nested, compare_state_dicts
from models.dataset_manifest import load_manifest, sha256_file
from transition_diagnostics import make_training_schedule, parameter_groups
from summarize_learning_transition import read_json, read_rows, require

write_json = prior.write_json
SOURCE = ROOT / 'results/step14-learning-rate/run01'
PREVIOUS = ROOT / 'results/step15-generalization/run01'
START, END = 9600, 3200
BRANCHES = ('A_keep64', 'B_expand256')
STEPS = [0, 800, 1600, 3200]
CONTROL_STEPS = [0, 3200]
BEAM_STEPS = [0, 3200]
COHORTS = ('original64', 'additional192', 'train60', 'val20')
HELDOUT = ('train60', 'val20')
RUNTIME_FIELDS = lr.RUNTIME_FIELDS


def schedule_metadata(schedule: dict) -> dict:
    return {key: schedule[key] for key in ('sample_ids', 'attempt_updates', 'batch_size', 'seed', 'batches_sha256')}


def settings(scale: float) -> dict:
    return {'attempt_updates': END, 'source_attempt': START, 'batch_size': 2, 'learning_rate': .0001,
        'adam_betas': [.9, .999], 'adam_eps': 1e-8, 'weight_decay': 0., 'seed': 42, 'grad_clip': 1.,
        'grad_scaler_init_scale': scale, 'precision': 'amp_fp16_ctc_fp32', 'ctc_reduction': 'mean',
        'ctc_zero_infinity': False, 'evaluation_precision': 'fp32_unpadded_B1', 'normal_steps': STEPS,
        'control_steps': CONTROL_STEPS, 'control_cohorts': list(HELDOUT), 'beam_steps': BEAM_STEPS,
        'beam_conditions': ['normal'], 'beam_width': 10, 'shuffle_seed': 42,
        'fresh_optimizer_per_branch': False, 'sampler': 'fresh local epoch-full-index-shuffle-v1',
        'early_stopping': False, 'validation_selection': False, 'torch_threads': 2,
        'cudnn_deterministic': True, 'cudnn_benchmark': False, 'tf32': False}


def ordered_ids(plan: dict) -> list[str]:
    return [sample_id for cohort in COHORTS for sample_id in plan['cohorts'][cohort]]


def zero_counters(ids: list[str]) -> dict:
    return {'optimizer_steps': 0, 'skipped_optimizer_steps': 0,
            'attempted_exposures': {i: 0 for i in ids}, 'successful_update_exposures': {i: 0 for i in ids}}


def provenance(plan: dict, key: str, step: int) -> dict:
    require(key in BRANCHES and step in STEPS, 'Unplanned checkpoint branch or step')
    return {'mode': 'smoke_only', 'purpose': 'train_diversity_diagnostic', 'branch': key,
        'manifest_sha256': plan['manifest_sha256'],
        'sample_ids': plan['cohorts']['original64'] if step == 0 else plan['branches'][key]['train_ids'],
        'planned_sample_ids': plan['branches'][key]['train_ids'],
        'source_state_sha256': plan['prior_state_sha256'], 'source_attempt': START,
        'cumulative_attempt': START+step, 'generalization_evaluation': True,
        'development_only': True, 'deployment': False}


def validate_training_ids(ids: list[str], plan: dict, key: str, records: dict) -> None:
    """No held-out clip, recording or transcript can enter an optimizer batch."""
    require(key in BRANCHES, 'Unplanned training branch')
    allowed = set(plan['branches'][key]['train_ids'])
    excluded = set(plan['selection']['excluded_recording_ids'])
    heldout_ids = [i for cohort in HELDOUT for i in plan['cohorts'][cohort]]
    heldout_texts = {normalize_transcript(records[i]['text']) for i in heldout_ids}
    require(len(ids) == len(set(ids)) == 2, 'Unique batch2 required')
    for sample_id in ids:
        require(sample_id in allowed and sample_id not in heldout_ids, 'Unselected training ID')
        row = records[sample_id]
        require(row['split'] == 'train' and row['recording_id'] not in excluded
                and normalize_transcript(row['text']) not in heldout_texts, 'Training/evaluation leakage')


def load_selected_cache(source: dict, plan: dict) -> dict:
    require(source['split'] in ('train', 'val'), 'Test cache access forbidden')
    require(source['id'] in ordered_ids(plan), 'Unselected cache access forbidden')
    return verify_cache(source)


def create_plan(out: Path, *, persist: bool = True) -> dict:
    require(out.is_relative_to(ROOT / 'results') and not out.exists(), 'New results folder required')
    p14, c14 = read_json(SOURCE / 'plan.json'), read_json(SOURCE / 'completed.json')
    b14 = read_json(SOURCE / 'B_reduce/completed.json')
    p15, c15 = read_json(PREVIOUS / 'plan.json'), read_json(PREVIOUS / 'completed.json')
    summary_path = ROOT / 'docs/단계별_개선/15_검증결과.json'
    summary, quality = read_json(summary_path), read_json(PREVIOUS / 'quality_checks.json')
    require(c14['status'] == b14['status'] == c15['status'] == 'completed'
            and summary['status'] == 'completed_and_verified' and quality['status'] == 'passed', 'Prior work incomplete')
    require(sha256_file(summary_path) == quality['artifacts'][summary_path.relative_to(ROOT).as_posix()], 'Stage15 summary changed')
    require(sha256_file(SOURCE / 'plan.json') == c14['plan_sha256'] == b14['plan_sha256'], 'Stage14 plan changed')
    require(sha256_file(SOURCE / 'B_reduce/completed.json') == c14['branch_completed_sha256']['B_reduce'], 'Source completion changed')
    require(sha256_file(PREVIOUS / 'plan.json') == c15['plan_sha256'] == summary['provenance']['plan_sha256']
            and sha256_file(PREVIOUS / 'completed.json') == summary['provenance']['completed_sha256'], 'Stage15 bindings changed')
    for path in (SOURCE/'plan.json', SOURCE/'completed.json', SOURCE/'B_reduce/completed.json'):
        require(sha256_file(path) == p15['prior_files'][path.relative_to(ROOT).as_posix()], 'Stage15-bound source lineage changed')
    state_path = SOURCE / 'B_reduce/training_state_9600.pt'
    state_spec = [row for row in b14['resume_states'] if row['step'] == START]
    require(len(state_spec) == 1 and sha256_file(state_path) == state_spec[0]['sha256'], 'Source full state changed')
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    checkpoint = SOURCE / 'B_reduce/step_9600.pt'
    require(sha256_file(checkpoint) == p15['models']['B_reduce9600']['checkpoint_sha256'], 'Source inference weights changed')
    require(compare_state_dicts(state['model_state_dict'], torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])['exact'],
            'Full state/checkpoint disagreement')
    source_contract = {'attempt': START, 'source_hashes': p14['source_hashes'],
        'settings': p14['branches']['B_reduce']['settings'], 'schedule': schedule_metadata(p14['schedule']),
        'provenance': {**lr.provenance(p14, 'B_reduce'), 'plan_sha256': b14['plan_sha256']},
        'counters': {key: b14[key] for key in ('optimizer_steps', 'skipped_optimizer_steps', 'attempted_exposures', 'successful_update_exposures')}}
    require(compare_nested(source_contract, {key: state[key] for key in source_contract})['exact'], 'Source state contract mismatch')
    environment = {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                   'numpy': np.__version__, 'python': sys.version}
    require(environment == c14['environment'], 'Execution environment changed')
    document, metadata = load_manifest(prior.MANIFEST, verify_files=False)
    require(metadata['manifest_sha256'] == prior.MANIFEST_SHA == p15['manifest_sha256'], 'Manifest changed')
    original = p14['cohorts']['expanded64']
    require(original == p15['training_ids'], 'Original training lineage mismatch')
    selection = select_diversity(document['records'], original, p15['selection']['cohorts'], total=256, seed=42)
    cohorts = {'original64': selection['training64'], 'additional192': selection['additional192'], **selection['heldout']}
    require([len(cohorts[c]) for c in COHORTS] == [64, 192, 60, 20], 'Fixed cohort sizes required')
    by_id = {row['id']: row for row in document['records']}
    training_settings = settings(state['scaler_state_dict']['scale'])
    branches = {key: {'train_ids': ids, 'settings': copy.deepcopy(training_settings),
                     'schedule': make_training_schedule(ids, attempt_updates=END, batch_size=2, seed=42)}
                for key, ids in zip(BRANCHES, (selection['training64'], selection['expanded256']))}
    for key, spec in branches.items():
        expected = 100 if key == 'A_keep64' else 25
        require(set(spec['schedule']['exposures'].values()) == {expected}, 'Unequal within-branch planned exposures')
    frozen = dict(p15['source_hashes'])
    for name, digest in frozen.items():
        require(sha256_file(ROOT / name) == digest, f'Frozen previous source changed: {name}')
    for name in ('scripts/diagnose_training_diversity.py', 'scripts/diversity_selection.py'):
        frozen[name] = sha256_file(ROOT / name)
    paths = [SOURCE/'plan.json', SOURCE/'completed.json', SOURCE/'B_reduce/completed.json', state_path, checkpoint,
             SOURCE/'B_reduce/step_9600/summary.json', SOURCE/'B_reduce/step_9600/rows.jsonl', SOURCE/'B_reduce/step_9600/posteriors.npz',
             PREVIOUS/'plan.json', PREVIOUS/'completed.json', PREVIOUS/'B_reduce9600/report.json',
             PREVIOUS/'B_reduce9600/rows.jsonl', PREVIOUS/'B_reduce9600/posteriors.npz', summary_path]
    prior14snapshot = read_json(SOURCE/'B_reduce/step_9600/summary.json')
    prior15report = read_json(PREVIOUS/'B_reduce9600/report.json')
    snapshot14 = [row for row in b14['snapshots'] if row['step'] == START]
    require(len(snapshot14) == 1 and sha256_file(SOURCE/'B_reduce/step_9600/summary.json') == snapshot14[0]['summary_sha256'],
            'Source snapshot summary changed')
    require(sha256_file(PREVIOUS/'B_reduce9600/report.json') == c15['models']['B_reduce9600'], 'Stage15 source report changed')
    for folder, report in ((SOURCE/'B_reduce/step_9600',prior14snapshot),(PREVIOUS/'B_reduce9600',prior15report)):
        for name, field in (('rows.jsonl','rows_sha256'),('posteriors.npz','posteriors_sha256')):
            require(sha256_file(folder/name)==report[field], 'Initial reproduction artifact changed')
    plan = {'schema': 'lipnet-training-diversity-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': str(prior.MANIFEST), 'manifest_sha256': prior.MANIFEST_SHA,
        'prior_state': str(state_path), 'prior_state_sha256': sha256_file(state_path), 'source_contract': source_contract,
        'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': sha256_file(checkpoint),
        'source_hashes': frozen, 'prior_files': {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths},
        'initial_counters': state['counters'], 'initial_scaler': state['scaler_state_dict'], 'environment': environment,
        'selection': selection, 'cohorts': cohorts, 'records': [by_id[i] for c in COHORTS for i in cohorts[c]],
        'settings': training_settings, 'branches': branches, 'steps': STEPS, 'resume_state_steps': STEPS,
        'source_attempt': START, 'additional_attempt_updates': END,
        'scope': {'training_performed': True, 'validation_cache_access': True, 'test_cache_access': False,
                  'development_evaluation': True, 'deployment': False, 'best_checkpoint_selection': False},
        'evaluation': {'cohort_order': list(COHORTS), 'normal_steps': STEPS, 'control_steps': CONTROL_STEPS,
            'control_cohorts': list(HELDOUT), 'beam_steps': BEAM_STEPS, 'beam_conditions': ['normal'],
            'expected_forwards': 3328, 'expected_beam_rows': 1344, 'primary_decoder': 'greedy'},
        'nonfinite_policy': 'Stop on nonfinite forward/loss/parameters or contract failure; GradScaler skips nonfinite gradients and all skips/exposures remain explicit. No metric-based early stop.',
        'interpretation': [
            'Optimizer/scaler/weights/RNG are identical full-state restores; only training examples and their fresh local sampler differ.',
            'State attempt/counters and schedule are LOCAL0..3200. Source attempt9600 and source counters remain bound separately; Adam step is not reset.',
            'Each branch restarts only its independent Python Random42 sampler, not dropout/global RNG.',
            'Equal3200 attempted batch2 updates and6400 presentations; repetitions100 vs25 and frame compute differ.',
            'additional192 is training only in B_expand256, an unseen diagnostic cohort in A_keep64.',
            'Both branches reproduce source outputs before either trains. Same336 normal evaluation and fixed heldout80 controls.',
            'Heldout train60 and all37 of its recordings remain reserved. val20 is previously used development data, not finaltest.',
            'No checkpoint selection by validation score, test50 access or webcam deployment.']}
    for key, spec in branches.items():
        for entry in spec['schedule']['batches']:
            validate_training_ids(entry['ids'], plan, key, by_id)
    if persist:
        out.mkdir(parents=True)
        write_json(out/'plan.json', plan)
    return plan


def new_objects():
    model = prior.LipNetBaseline().to('cuda').train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    return model, optimizer, scaler


def restore_exact(path, model, optimizer, scaler, contract):
    source = load_training_state(path, model=model, optimizer=optimizer, scaler=scaler, expected_contract=contract)
    audit = compare_nested({key: source[key] for key in RUNTIME_FIELDS}, lr.runtime_state(model, optimizer, scaler))
    require(audit['exact'] and all(g['lr'] == .0001 for g in optimizer.param_groups), 'Full runtime restore mismatch')
    return audit


@torch.inference_mode()
def evaluate_snapshot(model, plan, cache, records, key, step, folder, checkpoint_sha):
    require(key in BRANCHES and step in STEPS, 'Unplanned evaluation branch or step')
    folder.mkdir()
    started = time.perf_counter()
    rows, arrays = [], {}
    with prior.preserve_rng_and_mode(model):
        for cohort in COHORTS:
            for sample_id in plan['cohorts'][cohort]:
                source, item = records[sample_id], cache[sample_id]
                conditions = ['normal', 'static_middle', 'shuffle_42'] if step in CONTROL_STEPS and cohort in HELDOUT else ['normal']
                normal, normal_text = None, None
                for condition in conditions:
                    frames, transform = prior.transform_frames(item['frames'], 'shuffle' if condition == 'shuffle_42' else condition,
                                                               sample_id=sample_id, seed=42)
                    lp, _ = prior.forward(model, frames, capture=False)
                    greedy = prior.ctc_greedy_decode(lp)
                    beam = prior.ctc_beam_search_decode(lp, beam_width=10) if condition == 'normal' and step in BEAM_STEPS else None
                    diagnostics = prior.analyze_ctc(lp, item['label'], beam_text=beam if beam is not None else '')
                    if beam is None:
                        diagnostics = {k:v for k,v in diagnostics.items() if not k.startswith('beam')}
                    target = torch.tensor(item['label'], dtype=torch.long)
                    loss = torch.nn.functional.ctc_loss(torch.from_numpy(lp[:,None,:]), target,
                        torch.tensor([len(lp)]), torch.tensor([len(target)]), blank=0, reduction='mean', zero_infinity=False)
                    require(bool(torch.isfinite(loss)), 'Nonfinite evaluation CTC')
                    posterior_key = sample_id+'__'+condition
                    row = {'branch': key, 'step': step, 'cumulative_attempt': START+step,
                        'id': sample_id, 'split': source['split'], 'cohort': cohort, 'condition': condition,
                        'speaker_id': source['speaker_id'], 'recording_id': source['recording_id'],
                        'length_bin': length_bin(source['frames']), 'reference': source['text'], 'frames': source['frames'],
                        'greedy': prior.prediction_row(source,greedy), 'beam': prior.prediction_row(source,beam) if beam is not None else None,
                        'ctc': diagnostics, 'ctc_loss_per_target_token': float(loss), 'posterior_key': posterior_key,
                        'transform': transform, 'comparison_to_normal': None}
                    if condition == 'normal':
                        normal, normal_text = lp, greedy
                    else:
                        row['comparison_to_normal'] = prior.compare_posteriors(normal,lp,normal_text=normal_text,control_text=greedy)
                    rows.append(row)
                    arrays[posterior_key] = lp
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(folder/'posteriors.npz',**arrays)
    summary = {'branch':key,'step':step,'cumulative_attempt':START+step,'rows':len(rows),
        'checkpoint_sha256':checkpoint_sha,'rows_sha256':sha256_file(folder/'rows.jsonl'),
        'posteriors_sha256':sha256_file(folder/'posteriors.npz'),'seconds':time.perf_counter()-started,
        'conditions':['normal','static_middle','shuffle_42'] if step in CONTROL_STEPS else ['normal'],
        'control_cohorts':list(HELDOUT) if step in CONTROL_STEPS else [],'beam_performed':step in BEAM_STEPS}
    write_json(folder/'summary.json',summary)
    brief={c:{'cer':float(np.mean([r['greedy']['cer'] for r in rows if r['cohort']==c and r['condition']=='normal'])),
              'exact':sum(r['greedy']['exact'] for r in rows if r['cohort']==c and r['condition']=='normal')} for c in COHORTS}
    print(json.dumps({'evaluation':key,'local_step':step,'cohorts':brief,'seconds':summary['seconds']}),flush=True)
    return summary


def save_snapshot(folder, model, optimizer, scaler, plan, key, step, counters, cache, records, plan_sha):
    path = folder/f'step_{step:04d}.pt'
    model.save_checkpoint(str(path),training_data=provenance(plan,key,step),training_run={
        'diversity_diagnostic':True,'branch':key,'attempt_updates':step,'cumulative_attempt':START+step,
        'optimizer_steps':counters['optimizer_steps'],'skipped_optimizer_steps':counters['skipped_optimizer_steps'],
        'cumulative_optimizer_steps':plan['initial_counters']['optimizer_steps']+counters['optimizer_steps'],
        'cumulative_skipped_optimizer_steps':plan['initial_counters']['skipped_optimizer_steps']+counters['skipped_optimizer_steps'],
        'plan_sha256':plan_sha,'source_training_state_sha256':plan['prior_state_sha256'],
        'inference_only':True,'selection':'fixed_attempt_schedule','settings':plan['settings']})
    before = copy.deepcopy(lr.runtime_state(model,optimizer,scaler))
    snapshot_folder=folder/f'step_{step:04d}'
    summary=evaluate_snapshot(model,plan,cache,records,key,step,snapshot_folder,sha256_file(path))
    audit=compare_nested(before,lr.runtime_state(model,optimizer,scaler))
    require(audit['exact'],'Evaluation modified training runtime')
    write_json(snapshot_folder/'runtime_audit.json',audit)
    return {'step':step,'folder':snapshot_folder.name,'checkpoint':path.name,'checkpoint_sha256':sha256_file(path),
            'summary_sha256':sha256_file(snapshot_folder/'summary.json'),'runtime_audit_sha256':sha256_file(snapshot_folder/'runtime_audit.json')}


def save_state(folder, model, optimizer, scaler, plan, key, step, counters, plan_sha):
    path=folder/f'training_state_{step:04d}.pt'
    save_training_state(path,model=model,optimizer=optimizer,scaler=scaler,attempt=step,
        schedule=schedule_metadata(plan['branches'][key]['schedule']),
        provenance={**provenance(plan,key,step),'plan_sha256':plan_sha},
        source_hashes=plan['source_hashes'],settings=plan['settings'],counters=counters)
    return {'step':step,'file':path.name,'sha256':sha256_file(path)}


def compare_initial_outputs(folder: Path, plan: dict) -> dict:
    """Match source's 64 normal plus stage15's 80x3 heldout predictions/arrays."""
    new_rows = {(r['id'],r['condition']):r for r in read_rows(folder/'rows.jsonl')}
    total, maximum = 0, 0.
    with np.load(folder/'posteriors.npz',allow_pickle=False) as current:
        for source_folder, ids, normal_only in (
            (SOURCE/'B_reduce/step_9600',set(plan['cohorts']['original64']),True),
            (PREVIOUS/'B_reduce9600',set(plan['cohorts']['train60']+plan['cohorts']['val20']),False)):
            old_rows = [r for r in read_rows(source_folder/'rows.jsonl') if r['id'] in ids
                        and (not normal_only or r['condition']=='normal')]
            expected_count = len(ids) if normal_only else 3*len(ids)
            require(len(old_rows)==expected_count,'Incomplete source output grid')
            with np.load(source_folder/'posteriors.npz',allow_pickle=False) as previous:
                for old in old_rows:
                    new=new_rows[(old['id'],old['condition'])]
                    for field in ('reference','frames','greedy','beam','ctc','transform','comparison_to_normal'):
                        require(new[field]==old[field],f'Initial output mismatch: {old["id"]}/{field}')
                    require(np.isclose(new['ctc_loss_per_target_token'],old['ctc_loss_per_target_token'],rtol=2e-6,atol=2e-6),
                            'Initial CTC reproduction mismatch')
                    a,b=previous[old['posterior_key']],current[new['posterior_key']]
                    require(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),'Initial posterior mismatch')
                    maximum=max(maximum,float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
                    total+=1
    require(total==304,'Expected64+240 source reproduction checks')
    return {'status':'passed','prediction_rows_compared':total,'posterior_arrays_compared':total,
            'posterior_arrays_exact':True,'posterior_max_absolute_difference':maximum}


def compare_branch_outputs(a_folder: Path, b_folder: Path) -> dict:
    arows,brows=read_rows(a_folder/'rows.jsonl'),read_rows(b_folder/'rows.jsonl')
    require(len(arows)==len(brows)==496,'Expected initial496 outputs per branch')
    for a,b in zip(arows,brows):
        require({k:v for k,v in a.items() if k!='branch'}=={k:v for k,v in b.items() if k!='branch'},
                'Initial cross-branch prediction mismatch')
    with np.load(a_folder/'posteriors.npz',allow_pickle=False) as a,np.load(b_folder/'posteriors.npz',allow_pickle=False) as b:
        require(a.files==b.files and len(a.files)==496,'Initial cross-branch array inventory mismatch')
        for key in a.files:
            require(a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and np.array_equal(a[key],b[key]),
                    'Initial cross-branch posterior mismatch')
    return {'status':'passed','prediction_rows_exact':True,'posterior_arrays_exact':True,'posterior_arrays_compared':496}


def preflight(out, plan, cache, records):
    plan_sha=sha256_file(out/'plan.json')
    states,snapshots={},{}
    for key in BRANCHES:
        folder=out/key
        folder.mkdir()
        model,optimizer,scaler=new_objects()
        audit=restore_exact(plan['prior_state'],model,optimizer,scaler,plan['source_contract'])
        counters=zero_counters(plan['branches'][key]['train_ids'])
        states[key]=save_state(folder,model,optimizer,scaler,plan,key,0,counters,plan_sha)
        snapshots[key]=save_snapshot(folder,model,optimizer,scaler,plan,key,0,counters,cache,records,plan_sha)
        reproduction=compare_initial_outputs(folder/'step_0000',plan)
        write_json(folder/'restore_gate.json',{'status':'passed','source_runtime_exact':audit,
            'source_training_state_sha256':plan['prior_state_sha256'],'initial_training_state_sha256':states[key]['sha256'],
            'initial_outputs':reproduction,'stage_local_counters_reset_only':True,'optimizer_state_reset':False})
        del model,optimizer,scaler
        torch.cuda.empty_cache()
    a=torch.load(out/BRANCHES[0]/states[BRANCHES[0]]['file'],map_location='cpu',weights_only=True)
    b=torch.load(out/BRANCHES[1]/states[BRANCHES[1]]['file'],map_location='cpu',weights_only=True)
    cross=compare_nested({k:a[k] for k in RUNTIME_FIELDS},{k:b[k] for k in RUNTIME_FIELDS})
    require(cross['exact'],'Initial branch runtimes differ')
    outputs=compare_branch_outputs(out/BRANCHES[0]/'step_0000',out/BRANCHES[1]/'step_0000')
    write_json(out/'preflight_gate.json',{'status':'passed','plan_sha256':plan_sha,
        'checked_before_either_branch_training':True,'cross_branch_runtime_exact':cross,'cross_branch_initial_outputs':outputs,
        'restore_gate_sha256':{key:sha256_file(out/key/'restore_gate.json') for key in BRANCHES},
        'initial_states':states,'initial_snapshots':snapshots})
    print('PREFLIGHT PASSED: full source runtime restored twice;304 source outputs and496 cross-branch outputs exact',flush=True)
    return states,snapshots


def run_branch(out, plan, key, initial_state, initial_snapshot, cache, records):
    folder=out/key
    plan_sha=sha256_file(out/'plan.json')
    initial_path=folder/initial_state['file']
    require(sha256_file(initial_path)==initial_state['sha256'],'Prepared initial state changed')
    counters=zero_counters(plan['branches'][key]['train_ids'])
    contract={'attempt':0,'source_hashes':plan['source_hashes'],'settings':plan['settings'],
        'schedule':schedule_metadata(plan['branches'][key]['schedule']),
        'provenance':{**provenance(plan,key,0),'plan_sha256':plan_sha},'counters':counters}
    model,optimizer,scaler=new_objects()
    gate=restore_exact(initial_path,model,optimizer,scaler,contract)
    write_json(folder/'start_gate.json',{'status':'passed','prepared_runtime_exact':gate,
        'initial_training_state_sha256':initial_state['sha256'],'preflight_gate_sha256':sha256_file(out/'preflight_gate.json'),
        'next_local_attempt':1,'next_cumulative_attempt':START+1})
    groups=parameter_groups(model)
    exposures,successful=Counter(counters['attempted_exposures']),Counter(counters['successful_update_exposures'])
    snapshots,states=[initial_snapshot],[initial_state]
    compute={'raw_frame_presentations':0,'padded_frame_presentations':0}
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    with (folder/'telemetry.jsonl').open('x',encoding='utf-8') as stream:
        for entry in plan['branches'][key]['schedule']['batches']:
            validate_training_ids(entry['ids'],plan,key,records)
            result=prior.optimizer_attempt(model,optimizer,scaler,
                prior.collate([cache[i] for i in entry['ids']]),groups,torch.device('cuda'))
            counters['optimizer_steps']+=int(not result['skipped'])
            counters['skipped_optimizer_steps']+=int(result['skipped'])
            exposures.update(entry['ids'])
            if not result['skipped']:
                successful.update(entry['ids'])
            counters.update(attempted_exposures=dict(exposures),successful_update_exposures=dict(successful))
            frames=[records[i]['frames'] for i in entry['ids']]
            raw,padded=sum(frames),len(frames)*max(frames)
            compute['raw_frame_presentations']+=raw
            compute['padded_frame_presentations']+=padded
            result.update(entry)
            result.update(optimizer_steps=counters['optimizer_steps'],skipped_optimizer_steps=counters['skipped_optimizer_steps'],
                cumulative_attempt=START+entry['attempt'],
                cumulative_optimizer_steps=plan['initial_counters']['optimizer_steps']+counters['optimizer_steps'],
                cumulative_skipped_optimizer_steps=plan['initial_counters']['skipped_optimizer_steps']+counters['skipped_optimizer_steps'],
                frames=frames,target_lengths=[len(cache[i]['label']) for i in entry['ids']],
                raw_frame_presentations=raw,padded_frame_presentations=padded)
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            step=entry['attempt']
            if step in STEPS:
                snapshots.append(save_snapshot(folder,model,optimizer,scaler,plan,key,step,counters,cache,records,plan_sha))
                states.append(save_state(folder,model,optimizer,scaler,plan,key,step,counters,plan_sha))
            if step%100==0:
                print(json.dumps({'branch':key,'local_attempt':step,'cumulative_attempt':START+step,
                    'optimizer_steps':counters['optimizer_steps'],'skips':counters['skipped_optimizer_steps'],
                    'loss':result['loss'],'seconds':time.perf_counter()-started}),flush=True)
    require(dict(exposures)==plan['branches'][key]['schedule']['exposures'],'Final exposure mismatch')
    result={'status':'completed','branch':key,'plan_sha256':plan_sha,'attempt_updates':END,'cumulative_attempt':START+END,
        **counters,'cumulative_optimizer_steps':plan['initial_counters']['optimizer_steps']+counters['optimizer_steps'],
        'cumulative_skipped_optimizer_steps':plan['initial_counters']['skipped_optimizer_steps']+counters['skipped_optimizer_steps'],
        'snapshots':snapshots,'resume_states':states,'compute_totals':compute,'telemetry_sha256':sha256_file(folder/'telemetry.jsonl'),
        'restore_gate_sha256':sha256_file(folder/'restore_gate.json'),'start_gate_sha256':sha256_file(folder/'start_gate.json'),
        'seconds':time.perf_counter()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,**plan['scope']}
    write_json(folder/'completed.json',result)
    del model,optimizer,scaler
    torch.cuda.empty_cache()
    return result


def run(out: Path):
    require(torch.cuda.is_available(),'CUDA required')
    torch.set_num_threads(2)
    plan=create_plan(out)
    plan_sha=sha256_file(out/'plan.json')
    print(json.dumps({'plan_frozen':plan_sha,'source_state_sha256':plan['prior_state_sha256'],
        'cohort_counts':{k:len(v) for k,v in plan['cohorts'].items()},'additional_attempts_per_branch':END}),flush=True)
    started=time.perf_counter()
    records={r['id']:r for r in plan['records']}
    cache={i:load_selected_cache(records[i],plan) for i in ordered_ids(plan)}
    write_json(out/'cache_audit.json',{'plan_sha256':plan_sha,'cache_ids_opened':list(cache),
        'cache_sha256':{i:records[i]['cache_sha256'] for i in cache},'test_cache_access':False})
    states,snapshots=preflight(out,plan,cache,records)
    for key in BRANCHES:
        run_branch(out,plan,key,states[key],snapshots[key],cache,records)
    require(sha256_file(out/'plan.json')==plan_sha,'Plan changed during experiment')
    for name,digest in {**plan['source_hashes'],**plan['prior_files']}.items():
        require(sha256_file(ROOT/name)==digest,f'Frozen input changed: {name}')
    require(sha256_file(prior.MANIFEST)==plan['manifest_sha256'],'Manifest changed')
    forwards,beams=0,0
    for key in BRANCHES:
        for step in STEPS:
            summary=read_json(out/key/f'step_{step:04d}'/'summary.json')
            forwards+=summary['rows']
            beams+=len(ordered_ids(plan)) if summary['beam_performed'] else 0
    require(forwards==3328 and beams==1344,'Evaluation scope mismatch')
    write_json(out/'completed.json',{'status':'completed','plan_sha256':plan_sha,
        'preflight_gate_sha256':sha256_file(out/'preflight_gate.json'),
        'branch_completed_sha256':{key:sha256_file(out/key/'completed.json') for key in BRANCHES},
        'cache_audit_sha256':sha256_file(out/'cache_audit.json'),'cache_ids_opened':list(cache),
        'forwards':forwards,'beam_rows':beams,'seconds':time.perf_counter()-started,
        'environment':plan['environment'],**plan['scope']})
    print('STAGE16 COMPLETED',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/step16-diversity/run01')
    args=parser.parse_args()
    run(args.out.resolve())
