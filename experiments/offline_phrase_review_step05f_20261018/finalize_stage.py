"""Summarize the bounded 5f work, preserving pending phrase review."""
from __future__ import annotations

from pathlib import Path
from prepare_policy import BASE, ROOT, binding, load, save


def main() -> None:
    """Produce a review handoff; never promote candidates to training examples."""
    try:
        parent = load(BASE / 'effective_parent_inputs.json')
        answers = load(BASE / 'user_evidence/parent_answers.json')
        alignment = load(BASE / 'alignment/execution/alignment_results.json')
        media_path = Path(load(BASE / 'phrase_media_selected.json')['manifest']['path'])
        media = load(media_path)
        page = load(BASE / 'phrase_review_manifest.json')
        audit_names = ['source_preflight', 'policy_audit', 'source_plan_audit', 'media_audit',
                       'parent_page_audit', 'Q02_revision_audit', 'alignment_preparation_audit',
                       'parent_answer_audit', 'alignment_plan_preflight',
                       'alignment_execution_audit', 'phrase_media_r2_audit',
                       'phrase_page_audit', 'source_preservation_final']
        audits = []
        for name in audit_names:
            path = BASE / 'audit' / (name + '.json')
            record = load(path)
            assert record['status'] == 'passed', name
            audits.append(binding(path))
        assert len(parent['records']) == len(alignment['records']) == len(media['records']) == 8
        assert alignment['neural_calls'] == 8 and alignment['training_updates'] == 0
        assert answers['actual_answer_count'] == 9 and answers['confirmed_parent_count'] == 8
        assert all(not r['phrase_audio_boundary_approved'] and not r['phrase_visual_boundary_approved'] for r in media['records'])
        assert all(r['phrase_audio_boundary_status'] == 'pending' for r in page['records'])
        phrases = [{
            'review_id': r['review_id'], 'parent_id': r['parent_id'],
            'candidate_id': r['candidate_id'], 'text': r['text'],
            'duration_seconds': r['phrase_audio']['duration_seconds'],
            'audio': binding(Path(r['phrase_audio']['path'])),
            'audio_boundary_status': 'pending', 'visual_boundary_status': 'pending',
            'actual_split': None, 'actual_training_frames': None, 'ctc_feasible': None,
            'training_ready': False,
        } for r in media['records']]
        report_path = ROOT / 'docs/녹화형_단계학습_2026-10-12/05f_첫8문장_전사확인과_구절정렬.md'
        summary_path = BASE / 'results/stage5f_summary.json'
        summary = {
            'schema': 'offline-stage5f-summary-v1',
            'status': 'parent_review_and_phrase_proposals_complete_phrase_boundary_review_pending',
            'fixed_parents': 8, 'speakers': 8, 'actual_parent_answers': 9,
            'confirmed_parent_transcripts': 8, 'qualitatively_confirmed_parent_audio': 8,
            'original_parent_end_clipping_reports': 1, 'separately_confirmed_parent_revisions': 1,
            'Q02_revision': '+120ms; unchanged start, separately reviewed end; not measured boundary error',
            'canonical_text_changes': 0,
            'selected_parent_audio_samples_16k': parent['selected_audio_samples_16k'],
            'selected_parent_audio_duration_seconds': parent['selected_audio_duration_seconds'],
            'acoustic_annotation_forward_calls': 8,
            'acoustic_posterior_frames': sum(r['posterior_shape'][0] for r in alignment['records']),
            'acoustic_execution_seconds': alignment['elapsed_seconds'],
            'acoustic_greedy_parent_exact_matches': sum(r['acoustic_greedy_text'] == r['canonical_parent_text'] for r in alignment['records']),
            'acoustic_model_role': 'Offline data annotation only; not the lipreading student or its input',
            'phrase_proposals': phrases, 'unique_proposed_phrase_texts': len({r['text'] for r in phrases}),
            'phrase_proposal_duration_seconds': sum(r['phrase_audio']['samples'] for r in media['records']) / 16000,
            'phrase_audio_boundary_approvals': 0, 'phrase_visual_boundary_approvals': 0,
            'visual_student_forward_calls': 0, 'optimizer_updates': 0,
            'new_training_clips': 0, 'new_training_cache_files': 0,
            'new_original_downloads': 0, 'service_changes': 0,
            'producer_tar_payload_bytes': 2944839568,
            'successful_independent_tar_payload_bytes': 41944471,
            'failed_auditor_reconstructed_tar_payload_bytes': 31368859,
            'tar_byte_accounting_note': 'Third value is reconstruction, not a directly instrumented counter. Phrase review and Q02 repair read no new TAR payload.',
            'maximum_simultaneous_temporary_original_bytes': 437217504,
            'temporary_originals_remaining': 0,
            'prior_frozen_artifact_entries_preserved': 1910,
            'additional_stage1_artifact_entries_preserved': 10,
            'prior_scope_entries_preserved': 10,
            'focused_tests_passed_unique': 43,
            'test_count_note': '6 original media + 6 phrase media + 20 alignment (includes earlier 14) + 7 RIFF audit + 4 independent Viterbi; byte/trace/HTTP audits are additional checks, not duplicate test counts.',
            'actual_split': None, 'source_provisional': True, 'training_ready': False,
            'prior_5c_strict_gate': False,
            'prior_OOV_hold_and_pending_preserved': True,
            'sources': [binding(BASE / 'effective_parent_inputs.json'),
                        binding(BASE / 'user_evidence/parent_answers.json'),
                        binding(BASE / 'alignment/execution/alignment_results.json'),
                        binding(media_path), binding(BASE / 'phrase_review_manifest.json')],
            'independent_audits': audits,
            'interpretation_limits': [
                'Whole-parent approval does not approve the shorter phrase boundaries.',
                'CTC occupancy spans and acoustic greedy outputs are not ground-truth speech or mouth boundaries.',
                'All eight acoustic greedy transcripts differ from confirmed text; no labels were replaced by automatic outputs.',
                'Q03 and Q07 have whole-parent text, but the narrower CTC proposal still needs its own boundary review.',
                'Review videos are recompressed previews, not training pixels or precise articulation certification.',
                'Source-media technical integrity is separate from human speech boundary quality.',
            ],
        }
        save(summary_path, summary)
        table = '\n'.join(f'| {r["review_id"]} | {r["text"]} | {r["duration_seconds"]:.2f}초 | 미확인 | 미확인 |' for r in phrases)
        report = f'''# 녹화형 5f — 첫 8문장 확인과 구절 후보 준비

2026-10-18. **8명 각각 한 문장의 전체 전사·음성을 확인했고, 짧은 구절 8개의 자동 경계 후보를 준비했다. 구절 자체의 음성·입술 경계는 아직 미확인이다.** 새 학습이나 인식 성능 평가는 실행하지 않았다.

## 이번 결과

사용자 실제 답변은 9개다. Q01·Q03~Q08은 원래 전체 문장이 일치하고 잘림·혼입이 없다고 확인됐다. Q02는 원래 음성에서 마지막 ‘뻐’가 거의 들리지 않는다는 답변을 받았다. 시작은 그대로 두고 끝을 0.12초 늘린 별도 음성에서 문장 일치·마지막까지 온전함·혼입 없음이 확인됐다. 원본과 최초 잘림 답변을 보존했으며, 0.12초를 실제 경계 오차의 측정값으로 해석하지 않는다. 전사 텍스트 변경은 없다.

선택한 전체 음성은 446,549개 16kHz 샘플, 합계 27.9093125초다. 전체 문장 확인은 아래 짧은 구절의 확인과 구분한다.

| 번호 | 자동 구절 후보 | 길이 | 구절 음성 경계 | 입술 시간 경계 |
|---|---|---:|---|---|
{table}

총 8개·7종의 표현이며 길이 합계는 10.04초다. Q06·Q08은 서로 다른 화자의 같은 표현이다. 이는 아직 검수가 끝난 독립 학습 예시 8개라는 뜻이 아니다.

## 처리 방식과 근거

고정 Q01~Q08만 기존 D: TAR에 연결해 영상·음성·라벨 24개 멤버의 해시를 확인했다. 원본을 한 녹화씩 임시로 읽어 전체 문장 및 앞뒤 문맥의 검수 자료를 만들었다. 원본 임시 파일의 최대 동시 용량은 437,217,504바이트이고 완료 후 잔여는 0이다. 새 원본 다운로드는 없었다.

기존 로컬 `kresnik/wav2vec2-large-xlsr-korean`을 자료 주석용으로 8회 실행했다. 사람이 확인한 전체 문장을 강제 정렬해 1,390개의 음향 출력 프레임과 구절 시간 후보를 저장했다. 이 음향 모델은 학습 자료의 위치를 찾는 보조 도구이며, 최종 립리딩 프로그램의 입력과 학생 모델은 그대로다.

음향 출력의 CTC 점유 구간을 16kHz 정수 샘플 `320×t+40`으로 옮겼다. 원본 WAV의 48kHz 시작 샘플과 영상 보정값을 연결하고 영상 보정은 한 번 적용했다. Q02 수정 파일의 16,800은 문맥 파일 안에서의 시작 위치이므로, 실제 원본 녹화 원점 13,559,808/48,000=282.496초와 구분했다.

전체 문장 입력과 문맥 음성은 각각 원본에서 리샘플링했다. 구절 질문용 WAV는 저장된 문맥 PCM의 정확한 정수 구간을 사용하고, 정렬 입력의 같은 구간과 PCM 차이를 별도 기록했다. 검수 영상은 기존 25fps 미리보기에서 필요한 프레임을 잘라 만든 재압축 자료다. 영상 표시 구간이 음성 구간보다 조금 넓어질 수 있어 시간 차이를 기록했으며 학습 입력으로 사용하지 않는다.

## 검증과 한계

원본 PCM·독립 리샘플링·지역 파형 대응·844개 검수 격자/PTS/좌표·원래 검수 영상 16개 디코딩을 확인했다. 정렬 결과는 별도 FP64 알고리즘으로 전수 검산해 8개 경로·문자 구간이 일치했고 점수 최대 차이는 0이었다. 실제 9답변의 질문·음성 해시 연결, Q02의 원래 실패와 수정 확인 분리, 구절 미확인 상태 보존도 검증했다. 독립 미디어 감사는 원본 영상 픽셀을 다시 전수 디코딩한 검증과는 구분된다.

음향 모델의 자동 전사는 8문장 모두 사람이 확인한 전사와 다르다. 자동 출력을 정답으로 교체하지 않았다. 강제 정렬의 계산이 맞아도 발화나 입 움직임의 실제 경계가 맞는지는 별도 확인해야 한다. Q03·Q07은 후보 텍스트가 전체 문장과 같지만 자동 후보가 끝부분을 각각 약109.5ms·118.8125ms 제외하므로, 전체 문장을 확인했다는 이유로 짧은 후보를 승인하지 않는다.

구절 검수 자료의 독립 감사에서도 정확 PCM 16개·무음 영상 16개·표시 프레임 600개가 통과했다. 해당 8구절은 문맥 PCM과 별도 리샘플링한 정렬 입력의 같은 구간이 모두 일치했다. 최초 Q01 영상은 표기 33프레임과 실제 디코딩 32프레임이 달라 중단했다. 실패 자료를 남겨 두고, 같은 선택 프레임을 직접 디코딩한 뒤 25fps로 인코딩한 별도 r2에서 해결했다. 음성·구절·시간 후보는 바꾸지 않았다. 최초 실패의 세부 mux 원인까지 확정한 것은 아니다.

고유 테스트 43개와 별도 전수 감사가 통과했다. 기존 동결 항목 1,910개와 1단계 별도 10개·서비스 등 보호 범위 10개는 보존됐다. 최초 정책·검수 화면·미디어 기록은 그때의 미확인 상태를 담은 스냅샷으로 남겨 두며, 현재 상태는 `results/stage5f_summary.json`과 `results/next_step_handoff.json`을 따른다.

## 다음 작업

다음 5g는 이 8구절의 **정확히 잘라낸 음성을 먼저 청취**하고, 잘림이나 다른 말 혼입이 있으면 새 후보로 수정해 다시 확인하는 작업이다. 음성 확인 후 무음 구절 영상과 앞뒤 문맥 영상을 비교한다. 실제 답변이 없는 후보는 미확인 상태로 유지한다.

이번 단계의 학습 업데이트·시각 학생 모델 추론·새 학습 캐시는 모두 0이다. 실제 train/dev 분할은 미배정이고 `training_ready=false`, `provisional=true`다. 기존 5c 엄격 기준 미충족, OOV 보류, 보호 자료와 이전 미응답 상태도 유지한다. Q09~Q20 전체 추출이나 새 학습, 6단계 문장 확대는 자동 진행하지 않는다. 지금은 추가 다운로드나 코랩이 필요하지 않다.

검수 화면: [새 8구절 확인](http://127.0.0.1:8773/)

근거 폴더: `experiments/offline_phrase_review_step05f_20261018/`의 `user_evidence/`, `alignment/`, `audit/`, `phrase_review_manifest.json`, `results/`.
'''
        with report_path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(report)
        save(BASE / 'results/next_step_handoff.json', {
            'schema': 'offline-stage5f-next-step-handoff-v1',
            'current_status': summary['status'], 'next_step': '5g_phrase_audio_then_qualitative_visual_review',
            'summary': binding(summary_path), 'report': binding(report_path),
            'parent_answers': binding(BASE / 'user_evidence/parent_answers.json'),
            'effective_parents': binding(BASE / 'effective_parent_inputs.json'),
            'phrase_media_selection': binding(BASE / 'phrase_media_selected.json'),
            'phrase_questions': binding(BASE / 'phrase_review_manifest.json'),
            'review_url': 'http://127.0.0.1:8773/',
            'review_ids': [r['review_id'] for r in phrases],
            'first_question_batch': ['Q01', 'Q02', 'Q03'],
            'actual_phrase_answers': 0, 'actual_visual_answers': 0,
            'preserve_previous_frozen_files': True,
            'use_new_revision_for_later_answers_and_modified_boundaries': True,
            'do_not_promote_parent_review_to_phrase_review': True,
            'do_not_extract_Q09_Q20_automatically': True,
            'do_not_train_or_make_training_cache_automatically': True,
            'do_not_advance_to_stage6': True,
            'additional_download_or_colab_needed_now': False,
            'training_ready': False,
        })
        print('Created stage5f summary, report and pending phrase-review handoff')
    except Exception as exc:
        raise RuntimeError('Cannot finalize stage5f before all bounded technical checks pass') from exc


if __name__ == '__main__':
    main()
