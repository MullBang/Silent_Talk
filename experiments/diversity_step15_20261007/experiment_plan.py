"""Predeclare the bounded dataset comparison; this module never trains a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SEEDS = (42, 43)
COHORTS = ('train24', 'train72')
UPDATES = 3600
SALT = 'step15-schedule-v1:'
PRIOR = ROOT / 'experiments/multisentence_step12_20261007/diagnostic_plan.json'
PRIOR_SHA = '2fc10937d9c9a879e62839cac57bbe9337319dff80a3bdf71b5f344f1d701063'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def learning_rate(update):
    require(type(update) is int and 1 <= update <= UPDATES, 'Update must be one-based within the fixed budget')
    return .001 if update <= 2400 else .0003 if update <= 3000 else .0001


def make_schedule(records, seed):
    """Match speaker order across cohorts while exhausting each speaker's queue."""
    require(seed in SEEDS, 'Unknown seed')
    grouped = defaultdict(list)
    for row in records:
        grouped[row['speaker_id']].append(row['id'])
    require(len(records) in (24, 72) and len(grouped) == 8 and
            len({row['id'] for row in records}) == len(records), 'Expected distinct 24 or 72 records from eight speakers')
    size = len(records) // 8
    require(all(len(ids) == size for ids in grouped.values()), 'Unbalanced speaker population')
    stream = []
    for round_index in range(900):
        speakers = sorted(grouped, key=lambda speaker: (digest(f'{SALT}speakers:{seed}:{round_index}:{speaker}'), speaker))
        epoch, position = divmod(round_index, size)
        for speaker in speakers:
            ids = sorted(grouped[speaker], key=lambda sid: (digest(f'{SALT}clips:{seed}:{speaker}:{epoch}:{sid}'), sid))
            stream.append(ids[position])
    schedule = [stream[index:index+2] for index in range(0, len(stream), 2)]
    require(len(schedule) == UPDATES and Counter(stream) == Counter({row['id']: 7200//len(records) for row in records}),
            'Unbalanced presentations')
    return schedule


def make_run(records, seed, cohort, expected_initial):
    require(cohort in COHORTS and len(records) == int(cohort.removeprefix('train')), 'Cohort size mismatch')
    schedule = make_schedule(records, seed)
    rows = {row['id']: row for row in records}
    return {'run_id': f'seed{seed}_{cohort}', 'seed': seed, 'cohort': cohort, 'arm': 'B_feature',
            'initial_state_hashes_expected': expected_initial,
            'train_ids': [row['id'] for row in records], 'schedule': schedule,
            'updates': UPDATES, 'clips_per_update': 2, 'presentations': 7200,
            'presentations_per_clip': 7200//len(records), 'presentations_per_speaker': 900,
            'student_frames_presented': sum(rows[sid]['frames'] for batch in schedule for sid in batch),
            'learning_rate_segments': [{'start_update': 1, 'end_update': 2400, 'lr': .001},
                                       {'start_update': 2401, 'end_update': 3000, 'lr': .0003},
                                       {'start_update': 3001, 'end_update': 3600, 'lr': .0001}],
            'checkpoint_updates': [600, 1200, 1800, 2400, 3000, 3600],
            'normal_train_evaluation_updates': [2400, 3600],
            'normal_dev_evaluation_updates': [1200, 2400, 3000, 3600],
            'static_shuffled_dev_evaluation_updates': [3600],
            'historical_probe_normal_evaluation_updates': [3600],
            'final_comparison_update': 3600, 'planned_output_directory': str(BASE / 'future_runs' / f'seed{seed}' / cohort)}


def corpus_profile(rows, groups):
    """Describe the already selected strings; these counts cannot reselect data."""
    profiles, vocabularies = {}, {}
    for group, ids in groups.items():
        texts = [unicodedata.normalize('NFC', rows[sid]['text']) for sid in ids]
        tokens = [token for sid in ids for token in rows[sid]['tokens']]
        words = [word for text in texts for word in text.split()]
        syllables = [char for text in texts for char in text if '\uac00' <= char <= '\ud7a3']
        vocabularies[group] = {'tokens': set(tokens), 'words': set(words), 'syllables': set(syllables)}
        profiles[group] = {'utterances': len(ids), 'frames': sum(rows[sid]['frames'] for sid in ids),
                           'unique_token_ids': sorted(set(tokens)), 'unique_token_classes': len(set(tokens)),
                           'hangul_syllable_types': len(set(syllables)), 'whitespace_word_types': len(set(words)),
                           'whitespace_word_occurrences': len(words)}
    missing = {group: {kind: sorted(vocabularies['dev16'][kind]-vocabularies[group][kind])
                       for kind in ('tokens', 'words', 'syllables')} for group in COHORTS}
    return {'groups': profiles, 'dev16_values_absent_from_training': missing,
            'post_selection_description_only': True, 'used_to_select_or_tune': False,
            'words_are_surface_forms_not_dictionary_or_morphological_units': True,
            'unseen_word_does_not_imply_impossible_jamo_ctc_output': True}


def make_plan(data, prior):
    selection = data['selection']
    rows = {row['id']: row for row in selection['records']}
    require(len(rows) == len(selection['records']), 'Duplicate selected record')
    groups = {name: selection[name+'_ids'] for name in (*COHORTS, 'dev16')}
    require(len(set(groups['train24'])) == 24 and len(set(groups['train72'])) == 72 and
            len(set(groups['dev16'])) == 16 and set(groups['train24']) < set(groups['train72']) and
            not set(groups['train72']) & set(groups['dev16']), 'Invalid nested training/development split')
    require(set(rows) == set(groups['train72']+groups['dev16']), 'Record scope differs from selected groups')
    require(all(type(row['frames']) is int and row['frames'] > 0 for row in rows.values()), 'Invalid frame budget')
    speakers = {rows[sid]['speaker_id'] for sid in groups['train72']}
    require(Counter(rows[sid]['speaker_id'] for sid in groups['dev16']) == Counter({speaker: 2 for speaker in speakers}),
            'Development must include two records for each training speaker')
    probe_ids = data['pool_profile']['prior_probe_ids']
    pool = {row['id']: row for row in data['pool_profile']['records']}
    require(len(probe_ids) == len(set(probe_ids)) == 14 and set(probe_ids) <= set(pool) and
            not set(probe_ids).intersection(rows), 'Historical probe denominator or separation changed')
    probe_groups = {'current_known_speakers': [sid for sid in probe_ids if pool[sid]['speaker_id'] in speakers],
                    'current_unseen_speakers': [sid for sid in probe_ids if pool[sid]['speaker_id'] not in speakers]}
    runs = [make_run([rows[sid] for sid in groups[cohort]], seed, cohort,
                     prior['initial_state_hashes_expected'][str(seed)]['B_feature'])
            for seed in SEEDS for cohort in COHORTS]
    for seed in SEEDS:
        pair = [run for run in runs if run['seed'] == seed]
        speaker_sequences = [[[rows[sid]['speaker_id'] for sid in batch] for batch in run['schedule']] for run in pair]
        require(speaker_sequences[0] == speaker_sequences[1] and
                all(len(set(batch)) == 2 for batch in speaker_sequences[0]), 'Unmatched speaker order or repeated speaker batch')
    return {'schema': 'lipnet-step15-diversity-training-design-v1', 'status': 'prepared_not_trained',
            'purpose': 'Compare more unique utterances with fewer repetitions at a fixed number of optimizer updates.',
            'runs': runs, 'seeds': list(SEEDS), 'arm': 'B_feature', 'cohorts': groups,
            'post_selection_corpus_profile': corpus_profile(rows, groups),
            'speaker_sampling': {'salt': SALT, 'speakers_per_round': 8, 'rounds': 900,
                                 'paired_speaker_order': True, 'different_speakers_per_update': True,
                                 'primary_to_secondary_recording_ratio': '2:1 in each cohort'},
            'optimizer': {'name': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.,
                          'gradient_clip': 1., 'precision': 'FP32 GPU student and FP32 CPU CTC'},
            'loss': {'ctc': 1., 'feature': 1., 'feature_type': 'cached visual teacher features; no teacher forward'},
            'model': {'architecture': 'lipnet-compact-v1', 'source': 'existing self-designed student and feature adapter',
                      'fresh_initialization_each_run': True, 'resume_prior_trained_model': False,
                      'same_initial_state_within_seed': True, 'no_audio_at_inference': True},
            'runtime_reference_not_rechecked_on_gpu_in_step15': prior['runtime'],
            'evaluation': {'primary': 'Pooled character error rate on all dev16 at final update3600, including internal spaces.',
                           'decoder': 'Greedy raw token collapse then tolerant jamo composition preserving U+FFFD.',
                           'required_secondary': ['WER', 'exact sentence count', 'token error rate', 'invalid character count',
                                                  'per-speaker CER', 'train CER', 'static and shuffled dev CER',
                                                  'historical probe14 normal CER separated by current known/unseen speaker'],
                           'comparison': 'For each seed compute CER(train72 model)-CER(train24 model), then arithmetic mean of the two deltas.',
                           'selection': 'Always report final3600; intermediate checkpoints describe learning and never select the winner.',
                           'followup_expansion_gate': {'both_seed_deltas_strictly_below_zero': True,
                                                      'mean_delta_at_most_percentage_points': -5.,
                                                      'completed_runs': 4, 'inference_failures': 0,
                                                      'statistical_significance_claim': False,
                                                      'meaning': 'A practical predeclared signal for further research, not webcam readiness.'},
                           'missing_output_policy': 'Keep the fixed denominator. Save failure explicitly; no success gate if any failure.',
                           'historical_probe14_role': 'Secondary previously inspected diagnostic; original split=train; not untouched test.',
                           'historical_probe14_ids': probe_ids,
                           'historical_probe14_groups': probe_groups,
                           'known_speaker_baseline': 'For each dev speaker use the first control24 ID for that speaker; never choose by dev target or prediction.',
                           'dev_used_for_training_early_stopping_or_model_choice': False},
            'budget': {'runs': 4, 'updates_per_run': 3600, 'total_updates': 14400, 'total_presentations': 28800,
                       'student_frames_presented': sum(run['student_frames_presented'] for run in runs),
                       'checkpoint_count': 24, 'neural_evaluation_rows_planned': 824,
                       'evaluation_rows_breakdown': {'train_normal': 384, 'dev_normal': 256,
                                                      'dev_static_shuffled': 128, 'historical_probe_normal': 56},
                       'max_minutes_per_run_including_resume': 35, 'max_minutes_all_jobs': 150,
                       'minimum_free_disk_gib_before_training': 5,
                       'same_updates_not_same_frames_flops_or_wall_time': True,
                       'automatic_extension': False, 'original_datasets_or_checkpoints_deleted': False},
            'interruptions': {'nonfinite_loss_or_gradient': 'Stop and preserve full state; do not skip a clip or treat run as completed.',
                              'resource_failure_or_timeout': 'Stop, record incomplete scope, preserve checkpoints; do not select a shorter run as final.',
                              'resume': 'At most one resume from the same complete model, adapter, optimizer, RNG and next-update state within the original time/update budget.',
                              'restart_from_scratch_for_better_result': False},
            'step16_execution_requirements': [
                'Create a separate loader/executor bound to this design; old Step12 CLI does not implement this cohort or LR schedule.',
                'Verify selected cache hashes and decode only selected student and training feature arrays.',
                'Recheck GPU runtime and seeded complete initial-state hashes against reference before training.',
                'Implement learning_rate(update) before each optimizer update; resume uses next global update and optimizer LR.',
                'Verify LR boundaries and resumed next-update behavior before the four full runs.',
                'Freeze execution source, runtime, schedules and data bindings and independently preflight before any optimizer update.',
                'Report frame presentations, wall time and clip exposures; do not claim equal compute from equal update counts.'
            ],
            'limitations': ['Data quantity and per-clip repetitions change together at a fixed update count.',
                            'All participants were in older historical training pools; dev16 is current same-speaker recording-disjoint development.',
                            'Only B_feature is studied, so this does not estimate B versus A generalization.',
                            'The shared LR decay is a stabilization hypothesis, not an established fix or a separate LR ablation.',
                            'Fresh paired controls replace direct comparisons with Step12, whose speakers and recording mix differ.',
                            'No improvement, human transcript approval, or webcam accuracy is established by preparing data.'],
            'optimizer_updates_executed': 0, 'new_model_inferences': 0, 'teacher_forwards': 0,
            'service_weights_updated': False, 'downloads': 0, 'new_human_approvals_added': 0}


def validate_plan(plan, data, prior):
    expected = make_plan(data, prior)
    require(all(plan[key] == value for key, value in expected.items()), 'Design or schedule differs from the declared comparison')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'verify'))
    args = parser.parse_args()
    require(bind(PRIOR)['sha256'] == PRIOR_SHA, 'Step12 reference changed')
    data_path = BASE / 'data_plan.json'
    import data_plan
    data = data_plan.verify_plan()
    prior = read(PRIOR)
    output = BASE / 'training_design.json'
    code = [bind(BASE / name) for name in ('experiment_plan.py', 'test_experiment_plan.py')]
    if args.command == 'build':
        require(not output.exists(), 'Refusing to overwrite a prepared design')
        plan = {**make_plan(data, prior), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'data_plan': bind(data_path), 'prior_initialization_and_runtime_reference': bind(PRIOR), 'generator_sources': code}
        with output.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        plan = read(output)
        require(plan['data_plan'] == bind(data_path) and plan['prior_initialization_and_runtime_reference'] == bind(PRIOR)
                and plan['generator_sources'] == code, 'Changed plan evidence or code')
        validate_plan(plan, data, prior)
    print(json.dumps({'status': 'prepared_not_trained', 'training_design': bind(output), 'updates_executed': 0}))


if __name__ == '__main__':
    main()
