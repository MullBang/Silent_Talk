"""Preserve a Step8 progress snapshot without pretending pending listening is done."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(spec):
    if bind(spec['path'])['sha256'] != spec['sha256']:
        raise ValueError('Changed input: ' + spec['path'])


def main():
    previous = BASE.parent / 'transcript_step07_20261007/results/step07_summary.json'
    if bind(previous)['sha256'] != 'ec7456b3a5fc6a7a4837241d40f01958cdfbecc43d4a17b2ada00edde2ced9ae':
        raise ValueError('Changed Step7 summary')
    findings = read(BASE / 'difference_findings.json')
    for spec in findings['bindings'].values():
        verify(spec)
    neighbors = read(BASE / 'context_neighbors.json')
    for spec in neighbors['inputs'].values():
        verify(spec)
    verify(neighbors['code'])
    for spec in neighbors['label_files']:
        verify(spec)
    preflight = read(BASE / 'results/import_preflight.json')
    if preflight['status'] != 'passed':
        raise ValueError('Import preflight failed')
    verify(preflight['importer'])
    for spec in preflight['upstream'].values():
        verify(spec)
    audit_path = BASE / 'results/context_neighbors_independent_audit.json'
    audit = read(audit_path)
    if audit['status'] != 'passed':
        raise ValueError('Neighbor audit failed')
    labels_path = BASE.parent / 'transcript_step07_20261007/results/label_targets.json'
    labels = read(labels_path)
    targets = {r['id']: r for r in labels['records']}
    evidence = []
    for path in sorted((BASE / 'user_evidence').glob('*.json')):
        doc = read(path)
        target = targets[doc['id']]
        if doc['schema'] != 'lipnet-step08-scoped-chat-listening-evidence-v1' or doc['scope'] != 'local_repetition_presence_only':
            raise ValueError('Unsupported chat evidence scope')
        if doc['canonical_target'] != target['canonical_target'] or doc['confirmed_text'] not in target['canonical_target']:
            raise ValueError('Evidence/target mismatch')
        if hashlib.sha256(doc['canonical_target'].encode()).hexdigest() != doc['canonical_target_sha256']:
            raise ValueError('Evidence target checksum mismatch')
        for key in ('label_targets', 'source_plan', 'media_manifest', 'played_to_user'):
            verify(doc[key])
        if any(doc[k] for k in ('full_transcript_human_approved', 'human_boundary_approved', 'training_ready', 'labels_modified')):
            raise ValueError('A scoped reply cannot create whole-sentence approval')
        evidence.append({'binding': bind(path), 'id': doc['id'], 'actual_user_answer': doc['actual_user_answer'],
                         'scope': doc['scope'], 'action': doc['action']})
    asked = ['lip_J_1_F_02_C071_A_001_042', 'lip_J_1_F_02_C075_A_012_014', 'lip_J_1_F_04_C321_A_012_023']
    answered = {d['id'] for d in evidence}
    artifacts = ['difference_findings.json', 'difference_review.md', 'build_difference_findings.py',
                 'context_neighbor_check.py', 'context_neighbors.json', 'test_context_neighbor_check.py',
                 'import_review.py', 'test_import_review.py', 'audit_context_neighbors.py',
                 'results/import_preflight.json', 'results/context_neighbors_independent_audit.json',
                 'results/tests.log', 'summarize_step08.py']
    summary = {'schema': 'lipnet-step08-progress-summary-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'difference_analysis_and_import_ready_human_listening_in_progress',
               'previous_step_summary': bind(previous), 'artifacts': {a: bind(BASE / a) for a in artifacts},
               'report': bind(ROOT / 'docs/실행계획_2026-10-07/08_전사차이_분석과_청취기록_연결.md'),
               'cohort': {'records': 120, 'train': 96, 'val': 24, 'roles_preserved': True},
               'priority_text_difference_counts': findings['category_counts'],
               'priority_records': 35, 'neighbor_metadata_counts': neighbors['counts'],
               'new_actual_scoped_user_evidence': evidence, 'pending_scoped_questions': [i for i in asked if i not in answered],
               'scope_note': 'New chat evidence confirms only the named local fragments; it is not a complete transcript or boundary review.',
               'full_transcript_approvals_added': 0, 'boundary_approvals_added': 0, 'actual_ui_export_imports': 0,
               'import_preflight': 'passed', 'neighbor_independent_audit': 'passed',
               'tests': {'importer': 24, 'intervals': 4, 'total_passed': 28},
               'training_ready': False, 'student_training_updates': 0, 'service_weights_changed': False,
               'labels_modified': False, 'additional_asr_inference': False, 'dataset_downloaded': False,
               'next_actions': ['Record the two pending actual listening answers without expanding their scope.',
                                'Continue review of remaining repetition/word differences and boundary uncertainties.',
                                'Validate actual user-exported review JSON if supplied; preserve held rows and original split roles.',
                                'Declare the actual dataset inclusion and boundary policy before preparing a new training plan.']}
    output = BASE / 'results/step08_summary.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': summary['status'], 'scoped_replies': len(evidence),
                      'pending_questions': len(summary['pending_scoped_questions']), 'summary': bind(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
