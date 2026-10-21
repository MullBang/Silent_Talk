"""Join versioned targets and context diagnostics; never promote training data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'new_data_step06_20261007'
OUT = BASE / 'review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(text):
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
                           for c in unicodedata.normalize('NFC', text)).split())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        following = [i]
        for j, y in enumerate(b, 1):
            following.append(min(row[j] + 1, following[-1] + 1, row[j - 1] + (x != y)))
        row = following
    return row[-1]


def comparison(target, prediction):
    reference, hypothesis = normalize(target), normalize(prediction)
    r0, h0 = reference.replace(' ', ''), hypothesis.replace(' ', '')
    return {'normalized_reference': reference, 'normalized_prediction': hypothesis,
            'exact': reference == hypothesis, 'without_spaces_exact': r0 == h0,
            'edits': distance(reference, hypothesis), 'characters': len(reference),
            'without_spaces_edits': distance(r0, h0), 'without_spaces_characters': len(r0)}


def pooled(rows):
    counts = {key: sum(r[key] for r in rows) for key in
              ('exact', 'without_spaces_exact', 'edits', 'characters', 'without_spaces_edits', 'without_spaces_characters')}
    counts['cer_percent'] = 100 * counts['edits'] / counts['characters']
    counts['without_spaces_cer_percent'] = 100 * counts['without_spaces_edits'] / counts['without_spaces_characters']
    return counts


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    expected = {
        OLD / 'results/review_package.json': '22287f7f8c208446b0172a0600fd96ee37df451aafac15d09f6ce0e79e2d3d29',
        BASE / 'results/label_targets.json': 'e791f567ded1860e5baa5935232eb1d8bc32326a27b34e66f5392472143a18f6',
        BASE / 'results/context_asr_results.json': '4deb0a00df54edc97020ea5584c54583c408d93ee3a7c5c41ebc96533618c5dc',
        BASE / 'user_alternative_choice.json': '100da7dc7f3012bff2fc852e3dc1753d883042a22aa125cd44b1eb366539b931',
        BASE / 'label_plan.json': '428d6e1df2246d1bf681e1d73a0cb0184c654380347bd7c4824645fc6cdfa93f',
        BASE / 'context_plan.json': '0f382a327f046b2b9cb07b50b70c90ba269076d79df9f53c5a62df9ba1eb4c5f',
    }
    for path, digest in expected.items():
        require(sha(path) == digest, f'Changed upstream input: {path.name}')
    old_package = read(OLD / 'results/review_package.json')
    old_data_ref = old_package['data']
    require(sha(old_data_ref['path']) == old_data_ref['sha256'], 'Changed prior review data')
    previous = read(old_data_ref['path'])
    targets = read(BASE / 'results/label_targets.json')
    context = read(BASE / 'results/context_asr_results.json')
    audit_path = BASE / 'results/independent_stage07_audit.json'
    # Bind the audit if already produced; final summary will require it separately.
    ids = [r['id'] for r in previous['records']]
    require(ids == targets['ids'] == context['ids'] and len(set(ids)) == 120, 'Cohort mismatch')
    records = []
    for old, target, ctx in zip(previous['records'], targets['records'], context['records']):
        require(old['id'] == target['id'] == ctx['id'], 'Record ordering mismatch')
        require(old['reference'] == target['source_text'] == ctx['reference'], 'Original label mismatch')
        require(old['split'] == target['split'] == ctx['split'], 'Role changed')
        require(ctx['status'] == 'ok' and target['canonical_target'] is not None, 'Incomplete target/context')
        row = dict(old)
        row['canonical_target'] = target['canonical_target']
        row['canonical_target_sha256'] = hashlib.sha256(row['canonical_target'].encode('utf-8')).hexdigest()
        row['target_status'] = target['target_status']
        row['context_text'] = ctx['full_context_text']
        row['selected_text_diagnostic_only'] = ctx['selected_text']
        row['words'] = ctx['words']
        row['canonical_comparisons'] = {
            'baseline': comparison(row['canonical_target'], old['asr_text']),
            'full_context': comparison(row['canonical_target'], ctx['full_context_text']),
        }
        b, c = (row['canonical_comparisons'][k]['without_spaces_exact'] for k in ('baseline', 'full_context'))
        row['comparison_group'] = 'both_supported' if b and c else 'one_supported' if b or c else 'both_differ'
        row['comparison_scope'] = 'same_model_diagnostic_not_independent_verification'
        row['automatic_approval'] = False
        row['full_transcript_human_approved'] = False
        row['human_boundary_approved'] = False
        row['training_ready'] = False
        if target['selection_evidence']:
            row['alternative_choice_evidence'] = target['selection_evidence']['binding']
            row['user_note'] = '사용자 확인: ‘애쓰는’이 한 번 들림. 대안 선택만 확인했으며 전체 문장·경계의 승인 기록은 아닙니다.'
        # Media already bound to Step6 package. Reuse files without another copy.
        for key in ('audio', 'context_audio', 'mp4_audio', 'preview_video', 'face_context'):
            if row.get(key):
                require(row[key].startswith('../results/media/'), 'Unexpected prior media URL')
                row[key] = '../../new_data_step06_20261007/' + row[key][3:]
                require((OUT / row[key]).resolve().is_file(), 'Missing reused media')
        row['roi_samples'] = [dict(s, url='../../new_data_step06_20261007/' + s['url'][3:]) for s in old['roi_samples']]
        records.append(row)
    require(Counter(r['split'] for r in records) == {'train': 96, 'val': 24}, 'Split changed')
    counts = dict(Counter(r['comparison_group'] for r in records))
    diagnostic = {
        'first_word_starts_at_zero': sum(bool(r['words']) and r['words'][0]['start'] == 0 for r in context['records']),
        'first_word_excluded_by_midpoint': sum(bool(r['words']) and not r['words'][0]['selected_by_midpoint'] for r in context['records']),
        'selected_text_usable_as_labels': False,
        'word_time_cues_prove_clipping': False,
    }
    data = {'schema': 'lipnet-step07-review-data-v1',
            'bindings': {str(p.relative_to(BASE.parent)).replace('\\', '/'): bind(p) for p in expected},
            'media_sha256': previous['media_sha256'], 'records': records, 'group_counts': counts,
            'canonical_comparison_metrics': {k: pooled([r['canonical_comparisons'][k] for r in records]) for k in ('baseline', 'full_context')},
            'word_time_diagnostic': diagnostic,
            'review_order_rule': 'post_diagnostic_display_only: both_differ, one_supported, both_supported; original order within each',
            'training_ready': False, 'automatic_approvals': 0, 'full_transcript_human_approvals': 0}
    OUT.mkdir(exist_ok=True)
    save(OUT / 'review_data.json', data)
    template = (BASE / 'review_template.html').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (OUT / 'index.html').write_text(template.replace('__REVIEW_DATA__', payload), encoding='utf-8')
    package = {'schema': 'lipnet-step07-review-package-v1', 'inputs': data['bindings'],
               'builder': bind(Path(__file__)), 'template': bind(BASE / 'review_template.html'),
               'data': bind(OUT / 'review_data.json'), 'html': bind(OUT / 'index.html'),
               'group_counts': counts, 'canonical_comparison_metrics': data['canonical_comparison_metrics'],
               'word_time_diagnostic': diagnostic, 'training_ready': False, 'manual_approvals_added': 0}
    save(BASE / 'results/review_package.json', package)
    print(json.dumps({k: package[k] for k in ('group_counts', 'canonical_comparison_metrics', 'word_time_diagnostic')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
