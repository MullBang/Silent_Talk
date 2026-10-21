"""Record an explicitly supplied actual visual answer, or summarize recorded evidence."""
import argparse
from pathlib import Path

from prepare_q01_revision import BASE, bind, read, write


def record(args):
    batch_path = BASE / 'question_batches' / args.batch
    batch = read(batch_path)
    row = next(r for r in batch['records'] if r['review_id'] == args.qid)
    approved = args.assessment == 'no_obvious_clipping'
    if approved:
        assert args.answer == '뚜렷한 잘림은 보이지 않음'
    number = max(int(p.name.split('_')[0]) for p in (BASE / 'user_evidence').glob('*.json')) + 1
    output = BASE / 'user_evidence' / f'{number:03}_{args.qid}_visual_{args.assessment}.json'
    write(output, {'schema': 'offline-stage5g-actual-visual-answer-v1', 'review_id': args.qid,
                   'question_item_id': f'["request_user_input_async","{args.call_id}",{args.index}]',
                   'question': row['question'], 'answer': args.answer, 'source': 'actual_user_chat_reply',
                   'question_manifest': bind(batch_path), 'record': row['record'],
                   'viewing_aid': batch['viewing_aid'], 'previous_inconclusive_answer': batch['previous_inconclusive_answer'],
                   'interpretation': f'Item-specific qualitative visual assessment: {args.assessment}.',
                   'qualitative_visual_no_obvious_clipping': approved, 'phrase_visual_boundary_approved': approved,
                   'visual_assessment': args.assessment, 'legacy_approval_field_scope': 'qualitative preview inspection only',
                   'precise_articulatory_boundary_approved': False, 'original_training_pixels_approved': False,
                   'training_ready': False})
    print(output.name)


def summarize(args):
    summary = read(BASE / f'results/stage5g_summary_{args.previous}.json')
    all_paths = sorted((BASE / 'user_evidence').glob('*.json'))
    visual = [(p, read(p)) for p in all_paths if read(p)['schema'] == 'offline-stage5g-actual-visual-answer-v1']
    originals = {r['review_id']: r for r in read(BASE / 'question_batches/008_all8_visual.json')['records']}
    for row in summary['records']:
        candidates = [(p, e) for p, e in visual if e.get('review_id') == row['review_id']]
        if not candidates:
            continue
        path, proof = candidates[-1]
        assert proof['record'] == originals[row['review_id']]
        approved = proof['qualitative_visual_no_obvious_clipping']
        assessment = proof.get('visual_assessment', 'no_obvious_clipping' if approved else 'inconclusive')
        row.update({'visual_status': 'qualitative_' + assessment,
                    'visual_answer_evidence': bind(path), 'human_phrase_visual_boundary_approved': approved,
                    'qualitative_visual_no_obvious_clipping': approved,
                    'precise_articulatory_boundary_approved': False, 'training_ready': False})
    confirmed = sum(r['human_phrase_visual_boundary_approved'] for r in summary['records'])
    inconclusive = sum('inconclusive' in r['visual_status'] for r in summary['records'])
    summary.update({'status': f'in_progress_all8_audio_qualitative_visual_confirmed_{confirmed}_of_8',
                    'actual_visual_answers': len(visual),
                    'total_actual_audio_and_visual_answer_units': summary['actual_phrase_answers'] + len(visual),
                    'selected_phrase_visual_confirmed': confirmed, 'visual_inconclusive_items': inconclusive,
                    'evidence': [bind(p) for p in all_paths], 'latest_visual_answer': bind(visual[-1][0]),
                    'latest_question_batch': visual[-1][1]['question_manifest'], 'latest_question_batch_answered': True,
                    'next_question_batch': None, 'new_review_videos_this_revision': 0,
                    'next': 'Continue item-specific visual review for remaining items; if all eight are confirmed, finalize the bounded review handoff without automatic training or new source extraction.'})
    output = BASE / f'results/stage5g_summary_{args.version}.json'
    write(output, summary)
    print(f'{args.version}: {confirmed}/8 qualitative no-obvious-clipping, {inconclusive} inconclusive.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('record')
    for flag in ['qid', 'batch', 'call-id', 'answer', 'assessment']:
        p.add_argument('--' + flag, required=True)
    p.add_argument('--index', type=int, required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--previous', required=True)
    p.add_argument('--version', required=True)
    args = parser.parse_args()
    record(args) if args.command == 'record' else summarize(args)
