"""Update review status in a new UI version, preserving the verified viewing controls."""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from build_boundary_aid_r5 import PAGE
from prepare_q01_revision import BASE, bind, read, write


def main(version: str, next_id: str):
    assert version.startswith('r') and version[1:].isdigit() and int(version[1:]) >= 6
    summary_path = BASE / f'results/stage5g_summary_{version}.json'
    summary = read(summary_path)
    prototype_path = BASE / 'ui/r5/manifest.json'
    manifest = deepcopy(read(prototype_path))
    statuses = {r['review_id']: r for r in summary['records']}
    rows = manifest['records']
    assert next_id in statuses and len(rows) == len(statuses) == 8
    for row in rows:
        status = statuses[row['review_id']]
        assert row['media']['phrase_audio'] == status['selected_audio']
        assert status['human_phrase_audio_boundary_approved'] and not status['training_ready']
        proof = read(Path(status['visual_answer_evidence']['path']))
        assert bind(Path(status['visual_answer_evidence']['path'])) == status['visual_answer_evidence']
        approved = status['human_phrase_visual_boundary_approved']
        if approved:
            assert proof['review_id'] == row['review_id'] and proof['qualitative_visual_no_obvious_clipping']
            assert proof['record']['phrase_video'] == row['media']['phrase_video']
            assert proof['record']['context_video'] == row['media']['phrase_context_video']
        else:
            assert not proof['phrase_visual_boundary_approved']
        row.update({'state': '음성 확인 완료 · 뚜렷한 입술 잘림 보이지 않음' if approved else '음성 확인 완료 · 입술 경계 구분 어려움',
                    'phrase_visual_confirmed': approved, 'human_visual_status': status['visual_status'],
                    'human_visual_answer': proof['answer'], 'visual_answer_evidence': status['visual_answer_evidence'],
                    'precise_articulatory_boundary_approved': False, 'training_ready': False})
    count = sum(r['phrase_visual_confirmed'] for r in rows)
    assert count == summary['selected_phrase_visual_confirmed']
    index = next(i for i, r in enumerate(rows) if r['review_id'] == next_id)
    page = PAGE.replace('먼저 Q01의', f'현재 {next_id}의')
    page = page.replace('입술 경계: 8구절 모두 구분 어려움 · 확인 처리 0', f'입술 검수: 뚜렷한 잘림 없음 {count} · 판단 어려움 {8-count}')
    page = page.replace('<span class="qid" id="qid">Q01</span>', f'<span class="qid" id="qid">{next_id}</span>')
    page = page.replace('<div class="compare">', '<p id="itemReviewStatus" class="hint"></p><div class="compare">', 1)
    anchor = "$('phraseTitle').textContent=current.text;"
    assert page.count(anchor) == 1
    page = page.replace(anchor, anchor + "$('itemReviewStatus').textContent=current.state;")
    page = page.replace('choose(0).catch(report);', f"$('item').value='{index}';choose({index}).catch(report);")
    page = page.replace('__DATA__', json.dumps(rows, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c'))
    output = BASE / 'ui' / version
    output.mkdir(exist_ok=False)
    with (output / 'index.html').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(page)
    manifest['routes']['/'] = bind(output / 'index.html')
    manifest.update({'version': version, 'summary': bind(summary_path), 'builder': bind(Path(__file__)),
                     'template_builder': bind(BASE / 'build_boundary_aid_r5.py'), 'previous_ui': bind(prototype_path),
                     'actual_visual_evidence': summary['latest_visual_answer'], 'default_review_id': next_id,
                     'current_pointer_modified': True,
                     'new_visual_approvals': 0, 'qualitative_visual_confirmations_total': count,
                     'visual_inconclusive_total': 8-count,
                     'new_visual_approvals_note': 'UI production adds none; the count reflects separately recorded actual user answers.'})
    manifest['source_bindings'].extend([bind(summary_path), summary['latest_visual_answer']])
    for value in manifest['routes'].values():
        assert bind(Path(value['path'])) == value
    write(output / 'manifest.json', manifest)
    # The current pointer is the explicit mutable UI entry point.
    temporary = BASE / 'ui/current.next.tmp.json'
    temporary.write_text(json.dumps({'manifest': bind(output / 'manifest.json')}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(BASE / 'ui/current.json')
    print(f'{version}: qualitative visual {count}/8; default {next_id}; media and controls unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    parser.add_argument('next_id')
    args = parser.parse_args()
    main(args.version, args.next_id)
