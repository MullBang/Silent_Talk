"""Freeze the user-authorized, bounded confirmed-phrase learning comparison."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / 'experiments/offline_dataset_step03_20261013'
REVIEW = ROOT / 'experiments/offline_phrase_boundary_step05g_20261018'


def read(path: Path) -> Any:
    """Read a declared JSON source without changing it."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot read {path}') from exc


def ref(path: Path) -> dict:
    """Bind one exact file, using bounded-memory hashing."""
    try:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f'Cannot bind {path}') from exc


def save(path: Path, value: Any) -> None:
    """Create only a new experiment artifact, never overwrite history."""
    try:
        assert path.resolve().is_relative_to(BASE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot save {path}') from exc


def prepare() -> None:
    """Fix data roles, paired schedules, scope and metrics before training."""
    try:
        original = read(OLD / 'results/cache_handoff.json')
        new = read(BASE / 'data/cache_handoff.json')
        review = read(REVIEW / 'results/stage5g_summary_r8.json')
        rows = [dict(r, source_cohort='existing') for r in original['records'] if r['kind'] == 'phrase']
        additions = [dict(r, source_cohort='confirmed_addition') for r in new['records']]
        approved = {r['parent_id']: r for r in review['records'] if r['review_id'] in ('Q01', 'Q02')}
        held = [r for r in review['records'] if r['review_id'] not in ('Q01', 'Q02')]
        assert len(rows) == 15 and len(additions) == len(approved) == 2
        assert {r['parent_id'] for r in additions} == set(approved)
        overrides = {}
        for row in rows + additions:
            assert ref(Path(row['npz_path']))['sha256'] == row['npz_sha256']
            assert ref(Path(row['metadata_path']))['sha256'] == row['metadata_sha256']
            meta = read(Path(row['metadata_path']))
            assert meta['text'] == row['text'] and meta['T'] == row['T']
            assert meta['training_ready'] is False and meta['provisional'] is True
            assert meta['audio_included'] is False and meta['raw_padding_frames'] == 0
            assert meta['tokenizer'] == original['tokenizer']
            assert meta['ctc_feasible'] and meta['ctc_min_frames'] <= row['T']
            if row['source_cohort'] == 'confirmed_addition':
                evidence = approved[row['parent_id']]
                assert evidence['human_phrase_audio_boundary_approved'] and evidence['human_phrase_visual_boundary_approved']
                assert evidence['text'] == row['text']
                if meta['split'] != 'train':
                    overrides[row['clip_id']] = {'source_split': meta['split'], 'experimental_split': 'train'}
                row['split'] = 'train'
                row['actual_review_id'] = evidence['review_id']
                row['audio_confirmation'] = evidence['audio_confirmation']
                row['visual_answer_evidence'] = evidence['visual_answer_evidence']
            else:
                assert meta['split'] == row['split']
        train = sorted(r['clip_id'] for r in rows if r['split'] == 'train')
        dev = sorted(r['clip_id'] for r in rows if r['split'] == 'dev')
        add_ids = sorted(r['clip_id'] for r in additions)
        all_rows = rows + additions
        mapping = {r['clip_id']: r for r in all_rows}
        assert len(mapping) == 17 and len(train) == 12 and len(dev) == 3
        assert not ({r['parent_id'] for r in held} & {r['parent_id'] for r in all_rows})
        for field in ['parent_id', 'recording_group', 'text']:
            assert not ({mapping[i][field] for i in train + add_ids} & {mapping[i][field] for i in dev}), field
        for source in ['video', 'label']:
            train_hashes = {read(Path(mapping[i]['metadata_path']))['source_members'][source]['sha256'] for i in train + add_ids}
            dev_hashes = {read(Path(mapping[i]['metadata_path']))['source_members'][source]['sha256'] for i in dev}
            assert not train_hashes & dev_hashes, source
        data = {'schema': 'offline-stage5h-confirmed-data-plan-v1', 'records': all_rows,
                'arms': {'A12': train, 'B14': sorted(train + add_ids)}, 'dev': dev,
                'new2': add_ids, 'experimental_split_overrides': overrides,
                'source_handoffs': [ref(OLD / 'results/cache_handoff.json'), ref(BASE / 'data/cache_handoff.json')],
                'source_review': ref(REVIEW / 'results/stage5g_summary_r8.json'),
                'held': [{'review_id': r['review_id'], 'parent_id': r['parent_id'], 'reason': 'visual_inconclusive'} for r in held],
                'source_flags_unchanged': True, 'training_ready': False,
                'bounded_experiment_authorized': True,
                'interpretation': 'Qualitative review supports bounded experimental inclusion, not precise articulation certification or production readiness.',
                'leakage_checks': 'Train/dev parent, recording, phrase text and source video/label SHA disjoint. Inherited protection also checked in new cache preparation.',
                'tokenizer': original['tokenizer'], 'normalization': original['normalization']}
        save(BASE / 'data_plan.json', data)
        schedule_bindings = {'A12': {}, 'B14': {}}
        accounting = {}
        for seed in [42, 43]:
            rng = random.Random(seed)
            epochs = []
            for _ in range(280):
                order = sorted(train + add_ids)
                rng.shuffle(order)
                epochs.append(order)
            for arm in ['A12', 'B14']:
                schedule = []
                for epoch, order in enumerate(epochs[:280 if arm == 'A12' else 240]):
                    permitted = [i for i in order if i in data['arms'][arm]]
                    for index in range(0, len(permitted), 2):
                        n = len(schedule) + 1
                        ids = permitted[index:index + 2]
                        lengths = [mapping[i]['T'] for i in ids]
                        schedule.append({'update': n, 'epoch': epoch + 1, 'clip_ids': ids, 'lengths': lengths,
                            'lr': .001 if n <= 840 else .0003 if n <= 1050 else .0001,
                            'true_frames': sum(lengths), 'dense_frames': 2 * max(lengths)})
                count = Counter(i for r in schedule for i in r['clip_ids'])
                assert len(schedule) == 1680 and set(count) == set(data['arms'][arm])
                assert set(count.values()) == {280 if arm == 'A12' else 240}
                path = BASE / f'schedules/{arm}_seed{seed}.json'
                save(path, {'schema': 'offline-stage5h-paired-order-v1', 'seed': seed, 'arm': arm,
                    'rule': 'Local Random(seed) permutations of sorted14; A filters to12 for280 epochs, B uses240 full epochs; independent of model RNG.',
                    'updates': schedule})
                schedule_bindings[arm][str(seed)] = ref(path)
                accounting[f'{arm}_seed{seed}'] = {'updates': len(schedule), 'exposures': dict(count),
                    'true_frames': sum(r['true_frames'] for r in schedule), 'dense_frames': sum(r['dense_frames'] for r in schedule)}
        policy = {'schema': 'offline-stage5h-bounded-policy-v1', 'client_date': '2026-10-19',
            'actual_user_instruction': '그럼 검수는 필요한 만큼만 하고 애매한 자료는 보류한 뒤, 확인된 자료로 학습과 평가를 진행해보자',
            'authorization_interpretation': 'Explicitly authorizes this bounded learning/evaluation. Supersedes the prior automatic-training prohibition only for this new experiment.',
            'review_stop': 'No additional Q03-Q08 review or raw extraction. Preserve uncertainty and all old approval evidence.',
            'allowed_optimizer_ids': data['arms'], 'allowed_evaluation_ids': sorted(mapping),
            'allow_provisional_cache_for_controlled_experiment': True, 'source_approval_changed': False,
            'new_downloads': False, 'service_replacement': False, 'automatic_stage6': False,
            'data_plan': ref(BASE / 'data_plan.json')}
        save(BASE / 'controlled_pilot_policy.json', policy)
        code = [BASE / 'runner.py', BASE / 'prepare_experiment.py', ROOT / 'backend/models/baseline.py',
            ROOT / 'backend/models/tokenizer.py', ROOT / 'experiments/offline_comparison_step05_20261013/execution.py',
            ROOT / 'experiments/offline_training_step04_20261013/engine.py',
            ROOT / 'experiments/offline_training_step04_20261013/evaluation_contract.py']
        old_plan = read(ROOT / 'experiments/offline_phrase_only_step05c_20261018/execution_plan.json')
        plan = {'schema': 'offline-stage5h-confirmed-learning-plan-v1', 'client_date': '2026-10-19',
            'seeds': [42, 43], 'steps': 1680, 'evaluations': [840, 1680], 'batch_size': 2,
            'data_plan': ref(BASE / 'data_plan.json'), 'policy': ref(BASE / 'controlled_pilot_policy.json'),
            'runtime': old_plan['runtime'], 'schedules': schedule_bindings,
            'bindings': [ref(p) for p in code] + [ref(BASE / 'controlled_pilot_policy.json')],
            'planned_accounting': accounting, 'main_updates': 6720,
            'evaluation': {'midpoint': 'arm training clips, normal only', 'final': 'all17 x normal/static/shuffled; four runs',
                'cohorts': ['train_common12', 'new2', 'dev3'], 'planned_rows': 256,
                'primary': 'Fixed final dev3 normal NFC corpus CER; same reused speakers/internal exploratory set, not a final test.',
                'new2': 'A has never trained these examples; B has. Their change is fitting the additions, not a generalization result.',
                'selection': 'Report both seeds and arithmetic mean; no best checkpoint/seed selection or automatic extension.',
                'research_criterion': 'Both seeds lower final dev3 normal CER, mean decrease >=0.05 absolute. Exploratory only; no stage6 or deployment automatically.'},
            'limits': {'wall_seconds_per_run': 900, 'cuda_reserved_bytes': 6 * 1024**3},
            'caveats': ['14parents/13recordings/8speakers/13unique phrase texts; only C002/C004 gain examples.',
                'A280 vs B240 exposures per phrase, changed frame counts and batching; not a pure diversity-only causal comparison.',
                'Existing boundary cautions and source provisional flags remain. No independent final-webcam evaluation.']}
        save(BASE / 'plan.json', plan)
        print(json.dumps({'plan': ref(BASE / 'plan.json'), 'old_train': len(train), 'expanded_train': len(train + add_ids),
            'dev': len(dev), 'held': len(held), 'main_updates': 6720, 'evaluation_rows': 256}, ensure_ascii=False))
    except Exception as exc:
        raise RuntimeError('Plan preparation failed; preserve any created evidence and diagnose before retry') from exc


if __name__ == '__main__':
    prepare()
