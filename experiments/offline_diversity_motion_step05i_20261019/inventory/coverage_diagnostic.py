"""Describe frozen A14/B26 development vocabulary coverage without neural calls."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def read(path: str | Path) -> dict[str, Any]:
    """Read an explicitly bound local JSON input."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot read coverage input: {path}') from exc


def binding(path: str | Path) -> dict[str, Any]:
    """Bind the exact source file used in this descriptive diagnostic."""
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def units(record: dict[str, Any]) -> dict[str, list[Any]]:
    """Separate whitespace eojeols, Hangul syllables and 69-class target IDs."""
    text = unicodedata.normalize('NFC', record['text']).strip()
    metadata = read(record['metadata_path'])
    if binding(record['metadata_path'])['sha256'] != record['metadata_sha256']:
        raise ValueError(f"Bound metadata changed: {record['clip_id']}")
    if metadata['text'] != record['text']:
        raise ValueError('Metadata transcription differs')
    return {'eojeol': text.split(),
            'hangul_syllable': [char for char in text if '\uac00' <= char <= '\ud7a3'],
            'jamo_space_token_id': metadata['label']}


def coverage(values: list[Any], support: set[Any]) -> dict[str, Any]:
    """Count repeated occurrences and unique types separately."""
    unique = set(values)
    covered = [v for v in values if v in support]
    return {'occurrences': len(values), 'covered_occurrences': len(covered),
            'occurrence_coverage': len(covered)/len(values) if values else None,
            'unique_types': len(unique), 'covered_unique_types': len(unique & support),
            'unique_type_coverage': len(unique & support)/len(unique) if unique else None,
            'uncovered_types': sorted(unique-support), 'covered_types': sorted(unique & support)}


def main() -> None:
    """Create a new report while preserving the frozen experiment inventory."""
    try:
        source = HERE / 'verified_expansion.json'
        inventory = read(source)
        if inventory['status'] != 'passed':
            raise ValueError('Source inventory has not passed')
        mapping = {row['clip_id']: row for row in inventory['records']}
        extracted = {sid: units(row) for sid, row in mapping.items()}
        dimensions = ['eojeol', 'hangul_syllable', 'jamo_space_token_id']
        arm_support = {arm: {dimension: {v for sid in inventory['comparison_arms'][arm]
            for v in extracted[sid][dimension]} for dimension in dimensions} for arm in ['A14', 'B26']}
        rows = []
        for cohort in ['dev_phrase3', 'dev_sentence3']:
            ids = inventory['cohorts'][cohort]
            for arm in ['A14', 'B26']:
                per_clip = []
                for sid in ids:
                    per_clip.append({'clip_id': sid, 'text': mapping[sid]['text'],
                        'coverage': {dimension: coverage(extracted[sid][dimension], arm_support[arm][dimension]) for dimension in dimensions}})
                aggregated = {dimension: coverage([v for sid in ids for v in extracted[sid][dimension]], arm_support[arm][dimension]) for dimension in dimensions}
                rows.append({'cohort': cohort, 'arm': arm, 'coverage': aggregated, 'per_clip': per_clip})
        out = {'schema': 'offline-stage5i-posthoc-vocabulary-coverage-v1', 'status': 'completed_descriptive_only',
            'source_inventory': binding(source), 'diagnostic_code': binding(__file__),
            'timing_scope': 'Prepared after fixed training/data plans, before completion/results inspection; no selection or training change.',
            'support_sizes': {arm: {dimension: len(values) for dimension, values in item.items()} for arm, item in arm_support.items()},
            'rows': rows, 'new_neural_calls': 0, 'new_training_updates': 0,
            'limitations': ['Eojeols use exact NFC whitespace-delimited forms, without stemming or morphological normalization.',
                'Hangul syllables count precomposed Hangul characters only. Jamo-space token IDs are separate target symbols and do not imply syllable/word recognition.',
                'Phrase and full sentence development cohorts share the same three parents; never combine their counts as independent evaluation samples.',
                'Observed support measures potential exposure only, not learned visual recognition or a causal explanation of errors.',
                'The selected training data and budget were already frozen; this report does not select additional data or tune on development outputs.']}
        path = HERE / 'posthoc_coverage.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(out, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'support': out['support_sizes'], 'coverage': [{k:r[k] for k in ['cohort', 'arm', 'coverage']} for r in rows], 'path':str(path)},ensure_ascii=False,indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError('Coverage diagnostic failed; preserve existing output before retry') from exc


if __name__ == '__main__':
    main()
