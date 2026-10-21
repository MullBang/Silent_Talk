"""Bind the confirmed-source context expansion and static-training control."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / 'experiments/offline_confirmed_learning_step05h_20261019'
STAGE3 = ROOT / 'experiments/offline_dataset_step03_20261013'


def read(path: str | Path) -> dict:
    """Read an explicitly selected evidence file."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot read evidence {path}') from exc


def ref(path: str | Path) -> dict:
    """Bind a file without loading all its bytes into memory."""
    try:
        path = Path(path).resolve()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f'Cannot bind file {path}') from exc


def save(path: str | Path, value: dict) -> None:
    """Create a new local artifact and never replace past evidence."""
    try:
        path = Path(path).resolve()
        assert path.is_relative_to(BASE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, AssertionError) as exc:
        raise RuntimeError(f'Cannot create artifact {path}; preserve any partial output') from exc


def prepare() -> None:
    """Validate scope and bind the budget before any neural execution."""
    try:
        previous = read(OLD / 'data_plan.json')
        source = read(STAGE3 / 'results/cache_handoff.json')
        rows = [dict(r) for r in previous['records']]
        rows += [dict(r, source_cohort='confirmed_parent_full_sentence')
                 for r in source['records'] if r['kind'] == 'full_sentence']
        mapping = {r['clip_id']: r for r in rows}
        assert len(rows) == len(mapping) == 32
        for r in rows:
            assert ref(r['npz_path'])['sha256'] == r['npz_sha256']
            assert ref(r['metadata_path'])['sha256'] == r['metadata_sha256']
            m = read(r['metadata_path'])
            assert m['clip_id'] == r['clip_id'] and m['text'] == r['text'] and m['T'] == r['T']
            assert m['training_ready'] is False and m['provisional'] is True
            assert m['audio_included'] is False and m['raw_padding_frames'] == 0
            assert m['ctc_feasible'] and m['ctc_min_frames'] <= m['T']
            assert m['split'] == r['split'] and m['tokenizer'] == source['tokenizer']
        cohorts = {
            'train_phrase14': sorted(previous['arms']['B14']),
            'train_sentence12': sorted(r['clip_id'] for r in rows if r['kind'] == 'full_sentence' and r['split'] == 'train'),
            'dev_phrase3': sorted(previous['dev']),
            'dev_sentence3': sorted(r['clip_id'] for r in rows if r['kind'] == 'full_sentence' and r['split'] == 'dev'),
        }
        assert [len(v) for v in cohorts.values()] == [14, 12, 3, 3]
        assert len(set(i for ids in cohorts.values() for i in ids)) == 32
        expanded = sorted(cohorts['train_phrase14'] + cohorts['train_sentence12'])
        dev = cohorts['dev_phrase3'] + cohorts['dev_sentence3']
        held = previous['held']
        assert not {r['parent_id'] for r in held} & {r['parent_id'] for r in rows}
        assert {mapping[i]['parent_id'] for i in cohorts['train_sentence12']} < {mapping[i]['parent_id'] for i in cohorts['train_phrase14']}
        for field in ['parent_id', 'recording_group', 'text']:
            assert not {mapping[i][field] for i in expanded} & {mapping[i][field] for i in dev}, field
        for field in ['video', 'label']:
            train_sha = {read(mapping[i]['metadata_path'])['source_members'][field]['sha256'] for i in expanded}
            dev_sha = {read(mapping[i]['metadata_path'])['source_members'][field]['sha256'] for i in dev}
            assert not train_sha & dev_sha
        for i in expanded:
            for j in dev:
                a, b = mapping[i]['text'].replace(' ', ''), mapping[j]['text'].replace(' ', '')
                assert a not in b and b not in a, (i, j)
        # Existing exact source artifacts are immutable evidence, not new approvals.
        frozen = read(OLD / 'results/completion_manifest.json')
        for item in frozen['artifacts']:
            assert ref(item['path'])['sha256'] == item['sha256']
        inventory = BASE / 'inventory/verified_expansion.json'
        assert inventory.exists(), 'Independent inventory must finish before plan freeze'
        inventory_result = read(inventory)
        assert inventory_result['status'] == 'passed'
        assert {key: sorted(ids) for key, ids in inventory_result['cohorts'].items()} == cohorts
        assert {r['clip_id']: r['npz_sha256'] for r in inventory_result['records']} == {r['clip_id']: r['npz_sha256'] for r in rows}
        data = {
            'schema': 'offline-stage5i-diversity-motion-data-v1', 'records': rows,
            'arms': {'A14': cohorts['train_phrase14'], 'B26': expanded, 'C26_static': expanded},
            'cohorts': cohorts, 'experimental_split_overrides': {},
            'held': held, 'held_parent_ids': [r['parent_id'] for r in held],
            'source_handoffs': [ref(OLD / 'data_plan.json'), ref(STAGE3 / 'results/cache_handoff.json')],
            'inventory': ref(inventory), 'frozen_previous': ref(OLD / 'results/completion_manifest.json'),
            'tokenizer': source['tokenizer'], 'normalization': source['normalization'],
            'source_training_ready': False, 'source_flags_unchanged': True,
            'bounded_experiment_authorized': True,
            'full_sentence_evidence_scope': 'Actual earlier parent transcript/audio confirmations and original cache QC; no transfer of phrase visual-boundary approval to full sentences.',
            'no_new_review_questions': True,
        }
        save(BASE / 'data_plan.json', data)
        schedules = {arm: {} for arm in data['arms']}
        accounting = {}
        for seed in [42, 43]:
            rng = random.Random(seed)
            orders = []
            for _ in range(260):
                order = list(expanded)
                rng.shuffle(order)
                orders.append(order)
            built = {}
            for arm, ids in data['arms'].items():
                tasks = []
                epochs = 260 if arm == 'A14' else 140
                for epoch, order in enumerate(orders[:epochs], 1):
                    selected = [i for i in order if i in ids]
                    for start in range(0, len(selected), 2):
                        pair = selected[start:start+2]
                        step = len(tasks)+1
                        lengths = [mapping[i]['T'] for i in pair]
                        tasks.append({'update': step, 'epoch': epoch, 'clip_ids': pair, 'lengths': lengths,
                                      'lr': .001 if step <= 910 else .0003 if step <= 1365 else .0001,
                                      'true_frames': sum(lengths), 'dense_frames': 2*max(lengths)})
                exposures = Counter(i for t in tasks for i in t['clip_ids'])
                assert len(tasks) == 1820 and set(exposures.values()) == {epochs}
                built[arm] = tasks
                path = BASE / f'schedules/{arm}_seed{seed}.json'
                save(path, {'schema': 'offline-stage5i-paired-schedule-v1', 'seed': seed, 'arm': arm,
                            'rule': 'Local Random(seed), 260 permutations of sorted26; A filters14 for260 epochs; B/C use first140 full epochs. Identical B/C batch order.',
                            'updates': tasks})
                schedules[arm][str(seed)] = ref(path)
                accounting[f'{arm}_seed{seed}'] = {'updates': len(tasks), 'exposures': dict(exposures),
                    'true_frames': sum(t['true_frames'] for t in tasks), 'dense_frames': sum(t['dense_frames'] for t in tasks)}
            assert built['B26'] == built['C26_static']
        policy = {
            'schema': 'offline-stage5i-bounded-policy-v1', 'client_date': '2026-10-19',
            'actual_user_instruction': '다음 단계 진행해줘. 애매한 자료는 계속 보류하고, 표현 다양성을 늘리면서 모델이 실제 입술 움직임을 활용하는지 비교해보자.',
            'scope': 'New bounded comparison uses existing confirmed phrase14 and previously confirmed parent sentence12. Earlier automatic-expansion restriction is superseded for these exact sources only.',
            'held_review_ids': [r['review_id'] for r in held], 'allow_provisional_cache_for_controlled_experiment': True,
            'source_approval_changed': False, 'additional_reviews': 0, 'raw_extractions': 0,
            'new_downloads': False, 'service_replacement': False, 'no_auto_extension': True,
            'allowed_optimizer_ids': data['arms'], 'allowed_evaluation_ids': sorted(mapping),
            'data_plan': ref(BASE / 'data_plan.json'),
        }
        save(BASE / 'controlled_pilot_policy.json', policy)
        code = [BASE / 'runner.py', BASE / 'prepare_experiment.py', ROOT / 'backend/models/baseline.py',
                ROOT / 'backend/models/tokenizer.py', ROOT / 'experiments/offline_comparison_step05_20261013/execution.py',
                ROOT / 'experiments/offline_training_step04_20261013/engine.py',
                ROOT / 'experiments/offline_training_step04_20261013/evaluation_contract.py']
        plan = {
            'schema': 'offline-stage5i-diversity-motion-plan-v1', 'client_date': '2026-10-19',
            'seeds': [42,43], 'steps': 1820, 'evaluations': [910,1820], 'batch_size': 2,
            'data_plan': ref(BASE / 'data_plan.json'), 'policy': ref(BASE / 'controlled_pilot_policy.json'),
            'runtime': read(OLD / 'plan.json')['runtime'], 'schedules': schedules,
            'bindings': [ref(p) for p in code] + [ref(BASE / 'controlled_pilot_policy.json')],
            'planned_accounting': accounting, 'main_updates': 10920, 'maximum_preflight_updates': 5,
            'training_conditions': {'A14': 'normal', 'B26': 'normal', 'C26_static': 'static_mid'},
            'limits': {'wall_seconds_per_run': 1200, 'cuda_reserved_bytes': 6*1024**3},
            'evaluation': {
                'midpoint': 'arm training IDs, normal only (C moving-input transfer, not static training-view fit)',
                'final': 'all32 x normal/static_mid/shuffled/reversed, six runs, fixed final1820',
                'planned_rows': 900, 'cohorts': list(cohorts),
                'primary': 'Dev phrase3 normal NFC corpus CER, report both seeds and arithmetic mean. Compare B-A for context expansion, C-B for motion training advantage.',
                'secondary': 'Dev full sentence3 scored separately; same parents as phrase3, not six independent examples.',
                'motion': 'Within B, ablatedCER-normalCER for static,shuffle,reverse; alongside Cnormal-Bnormal. Positive gap favors moving input but OOD sensitivity is not itself proof of useful motion reading.',
                'selection': 'No best seed/checkpoint selection; descriptive effects only, no statistical or deployment success criterion; no automatic extension.',
            },
            'caveats': [
                'Expansion changes 13 to24 exact texts and35 to76 word types within same14 parents/13 recordings/8 speakers; overlapping clips are not independent.',
                'A260 vsB/C140 exposures perclip; A/B differ in context,length,frame count,batches and expression support, not pure causal diversity.',
                'B/C share exactly same schedule, true lengths, initialization and update count; static middle retains appearance and duration cues.',
                'Dev3 parents reused internal development, existing speakers, no independent final/webcam test.',
                'Full sentences have earlier actual audio/transcript confirmation and source QC, no new precise visual-boundary certification.',
            ],
        }
        save(BASE / 'plan.json', plan)
        print(json.dumps({'plan': ref(BASE / 'plan.json'), 'main_updates': 10920, 'evaluation_rows': 900,
                          'training_clips': 26, 'held': len(held), 'new_raw_reads': 0}, ensure_ascii=False))
    except Exception as exc:
        raise RuntimeError("Preparation failed; preserve created evidence before retry") from exc


if __name__ == '__main__':
    prepare()
