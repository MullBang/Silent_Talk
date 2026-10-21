"""Post-hoc descriptive output diagnostics; no model execution or causal claims."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def describe(rows):
    raw = [token for row in rows for token in row['raw_argmax_ids']]
    characters = Counter(c for row in rows for c in row['prediction'] if not c.isspace())
    count = sum(characters.values())
    max_runs = [max((len(list(group)) for char, group in itertools.groupby(row['prediction']) if not char.isspace()), default=0) for row in rows]
    return {'sentences': len(rows), 'empty_predictions': sum(not row['prediction'] for row in rows),
            'unique_predictions': len({row['prediction'] for row in rows}),
            'frames': len(raw), 'blank_argmax_frames': raw.count(0), 'blank_argmax_percent': 100 * raw.count(0) / len(raw),
            'mean_prediction_characters_including_spaces': sum(len(row['prediction']) for row in rows) / len(rows),
            'mean_reference_characters_including_spaces': sum(len(row['reference']) for row in rows) / len(rows),
            'nonspace_prediction_characters': count, 'ah_characters': characters['아'],
            'ah_percent_of_nonspace_prediction_characters': 100 * characters['아'] / count if count else 0,
            'replacement_characters': characters['\ufffd'], 'top10_nonspace_prediction_characters': characters.most_common(10),
            'maximum_same_character_run': max(max_runs),
            'sentences_with_same_character_run_at_least5': sum(length >= 5 for length in max_runs)}


def main():
    models, refs = [], []
    for seed in (42, 43):
        for variant in ('control_old441', 'expanded510'):
            path = BASE / f'results/evaluation_seed{seed}_{variant}.json'
            doc = json.loads(path.read_text(encoding='utf-8'))
            if doc['status'] != 'completed' or (doc['seed'], doc['variant']) != (seed, variant):
                raise ValueError('Wrong evaluation')
            groups = {}
            for name in ('old71', 'new24'):
                cohort = doc['cohorts'][name]
                normal = cohort['conditions']['normal']['records']
                stats = {condition: describe(value['records']) for condition, value in cohort['conditions'].items()}
                unchanged = {}
                for condition in ('static', 'shuffled'):
                    perturbed = cohort['conditions'][condition]['records']
                    if [r['id'] for r in normal] != [r['id'] for r in perturbed]:
                        raise ValueError('Unpaired conditions')
                    unchanged[condition] = sum(a['prediction'] == b['prediction'] for a, b in zip(normal, perturbed))
                groups[name] = {'conditions': stats, 'identical_prediction_to_normal': unchanged,
                                'first3_examples_in_frozen_order': [{k: row[k] for k in ('id', 'reference', 'prediction')} for row in normal[:3]]}
            refs.append(bind(path)); models.append({'seed': seed, 'variant': variant, 'groups': groups})
    value = {'schema': 'lipnet-step10-posthoc-output-diagnostics-v1', 'status': 'completed',
             'code': bind(Path(__file__)), 'evaluations': refs, 'models': models,
             'posthoc': True, 'model_executed': False, 'labels_changed': False, 'training_changed': False,
             'limits': ['Descriptive saved-output counts, selected after observing worsened CER; not a predeclared outcome or proof of cause.',
                        'Blank argmax fraction is not a posterior blank probability.',
                        'Character repetition and temporal invariance do not independently identify the failing layer or loss term.']}
    path = BASE / 'results/output_diagnostics.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'path': str(path), 'models': [{**{k: m[k] for k in ('seed', 'variant')},
          'normal': {g: {k: x['conditions']['normal'][k] for k in ('ah_percent_of_nonspace_prediction_characters', 'maximum_same_character_run', 'sentences_with_same_character_run_at_least5', 'blank_argmax_percent')} for g, x in m['groups'].items()}}
          for m in models]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
