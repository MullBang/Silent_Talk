"""Independently inspect saved features and frame mappings without a model or producer imports."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(ref):
    require(bind(ref['path']) == {'path': ref['path'], 'sha256': ref['sha256']}, 'Changed binding: ' + ref['path'])


def check_arrays(features, student, native, gather, input_student, input_native, input_gather):
    require(features.dtype == np.float32 and features.shape == (len(student), 512), 'Feature dtype/shape')
    require(np.isfinite(features).all() and np.all(np.linalg.norm(features.astype(np.float64), axis=1) > 1e-8), 'Empty/nonfinite features')
    for x, expected in ((student, input_student), (native, input_native), (gather, input_gather)):
        require(x.dtype == np.int64 and x.ndim == 1 and np.array_equal(x, expected), 'Frame identity mismatch')
    require(len(student) == len(gather) and np.all(gather >= 0) and np.all(gather < len(native)), 'Invalid gather range')
    require(np.array_equal(native[gather], student), 'Gather source identity differs')


def main():
    plan_path = BASE / 'teacher_execution_plan.json'
    result_path = BASE / 'results/teacher_manifest.json'
    plan, result = read(plan_path), read(result_path)
    data, inputs = read(plan['data_plan']['path']), read(plan['input_manifest']['path'])
    require(result['status'] == 'completed' and result['teacher_plan'] == bind(plan_path), 'Incomplete teacher result')
    for key in ('data_plan', 'input_manifest', 'input_execution_plan', 'input_audit', 'checkpoint'):
        verify(plan[key])
        require(result[key] == plan[key], 'Result/plan mismatch: ' + key)
    for ref in plan['code'] + plan['input_files']:
        verify(ref)
    verify(result['progress'])
    progress = read(result['progress']['path'])
    rows, items = ({r['id']: r for r in doc['records']} for doc in (data, inputs))
    selected = [r['id'] for r in data['records'] if r['split'] == 'train' and r['comparison_group'] == 'both_supported']
    require(len(selected) == 69 and selected == data['teacher_candidate_ids'] == plan['selected_ids'] == result['ids'] ==
            [r['id'] for r in result['records']] == [r['id'] for r in progress['records']], 'Cohort differs')
    require(progress['teacher_plan'] == bind(plan_path), 'Progress plan mismatch')
    require({p.stem for p in (BASE / 'features').glob('*.npz')} == set(selected), 'Forbidden or missing feature file')
    require({p.stem for p in (BASE / 'teacher_results').glob('*.json')} == set(selected), 'Forbidden or missing record file')
    prior = read(BASE.parent / 'local_dataset_step04_20261007/teacher_results/train_manifest.json')
    require(bind(BASE.parent / 'local_dataset_step04_20261007/teacher_results/train_manifest.json')['sha256'] ==
            '126870d275a2a3850468e7b1a9f3b27daf0e26cacfa2168638fd6404d621aba3', 'Changed previous features')
    state = prior['state_sha256_before']
    require(result['state_sha256_before'] == result['state_sha256_after'] == state, 'Teacher state differs from earlier extraction')
    checked, max_mean_delta, max_std_delta = [], 0., 0.
    for record, saved in zip(result['records'], progress['records']):
        sid = record['id']
        row, item = rows[sid], items[sid]
        require(row['teacher_candidate'] is True and row['split'] == record['split'] == 'train', 'Teacher outside selected train')
        require(record['input_record'] == item and record['feature_file'] == saved['feature_file'], 'Input/output link mismatch')
        verify(saved['feature_file']); verify(saved['record_file'])
        require(read(saved['record_file']['path']) == record, 'Manifest/sidecar differs')
        input_ref = {'path': item['npz_path'], 'sha256': item['npz_sha256']}
        verify(input_ref)
        with np.load(input_ref['path'], allow_pickle=False) as incoming, np.load(record['feature_file']['path'], allow_pickle=False) as feature:
            require(set(feature.files) == {'features', 'student_source_indices', 'teacher_source_indices', 'gather_indices', 'metadata_json'}, 'Unexpected feature arrays')
            meta = json.loads(str(feature['metadata_json'].item()))
            require(record == {**meta, 'feature_file': saved['feature_file'], 'status': 'completed'}, 'Feature metadata differs')
            require(meta['data_record_sha256'] == hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(), 'Logical row checksum')
            values = feature['features']
            check_arrays(values, feature['student_source_indices'], feature['teacher_source_indices'], feature['gather_indices'],
                         incoming['student_source_indices'], incoming['teacher_source_indices'], incoming['student_teacher_rows'])
            require(len(values) == row['frames'] and meta['native_feature_shape'] == [1, len(incoming['teacher_bgr']), 512]
                    and meta['aligned_feature_shape'] == [row['frames'], 512], 'Frame/feature dimensions')
            require(meta['feature_values_sha256'] == hashlib.sha256(values.tobytes()).hexdigest(), 'Feature value checksum')
            norm = meta['normalization']
            pixels = incoming['teacher_bgr'].astype(np.float64)
            mean_delta, std_delta = abs(float(pixels.mean()) - norm['raw_mean']), abs(float(pixels.std(ddof=1)) - norm['raw_std'])
            require(norm['channel_order'] == 'BGR' and norm['std_correction'] == 1 and norm['dtype'] == 'float32', 'Wrong input normalization')
            require(mean_delta < .001 and std_delta < .001 and abs(norm['normalized_mean']) < .001 and abs(norm['normalized_std'] - 1) < .001, 'Whole-window mean/std mismatch')
            max_mean_delta, max_std_delta = max(max_mean_delta, mean_delta), max(max_std_delta, std_delta)
            for key in ('data_plan', 'input_manifest', 'input_execution_plan', 'input_audit', 'checkpoint', 'runtime'):
                require(meta[key] == plan[key], 'Record/plan mismatch: ' + key)
            require(meta['teacher_plan'] == bind(plan_path) and meta['state_sha256_before'] == meta['state_sha256_after'] == state, 'Record state/plan')
            require(meta['optimization_steps'] == meta['trainable_parameters'] == 0 and
                    all(meta[k] is False for k in ('training_ready', 'audio_used', 'transcript_used_for_teacher_forward', 'decoder_instantiated', 'optimizer_instantiated')), 'Unexpected model activity')
            if sid == selected[0]:
                repeat = meta['warm_repeat']
                require(repeat['bitwise_equal'] is True and repeat['max_absolute_difference'] == 0 and repeat['same_input_sha256'] == item['npz_sha256'], 'Repeat check missing')
            else:
                require(meta['warm_repeat'] is None, 'Unexpected repeat claim')
            checked.append({'id': sid, 'feature_file': saved['feature_file'], 'student_frames': len(values),
                            'teacher_frames': len(incoming['teacher_bgr']), 'input_mean_delta_float64_vs_saved_float32': mean_delta,
                            'input_std_delta_float64_vs_saved_float32': std_delta})
    require(sum(r['student_frames'] for r in checked) == result['student_frames'] == 8455, 'Aligned frame total')
    require(sum(r['teacher_frames'] for r in checked) == result['native_frames'], 'Native frame total')
    audit = {'schema': 'lipnet-step09-independent-feature-audit-v1', 'status': 'passed',
             'code': bind(Path(__file__)), 'data_plan': plan['data_plan'], 'teacher_plan': bind(plan_path),
             'teacher_manifest': bind(result_path), 'prior_feature_manifest': bind(BASE.parent / 'local_dataset_step04_20261007/teacher_results/train_manifest.json'),
             'record_count': 69, 'student_frames': 8455, 'native_frames': result['native_frames'],
             'max_raw_mean_delta': max_mean_delta, 'max_raw_std_delta': max_std_delta, 'records': checked,
             'forbidden_held_or_development_feature_files': 0,
             'limits': ['Independent saved-array and mapping audit; it does not independently rerun the GPU network.',
                        'The producer performed first-input exact repeat and frozen-state checks; saved claims are bound and checked here.',
                        'Input statistics are independently computed in float64 with .001 tolerance against saved float32 reductions.']}
    path = BASE / 'results/feature_audit.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k: audit[k] for k in ('status', 'record_count', 'student_frames', 'native_frames', 'max_raw_mean_delta', 'max_raw_std_delta')}))


if __name__ == '__main__':
    main()
