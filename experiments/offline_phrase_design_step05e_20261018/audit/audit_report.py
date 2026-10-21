"""Bind final report, actual numeric sources and the unexecuted8-parent handoff."""
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from audit_sources import ROOT, BASE, HERE, S3, read, binding, check


def inspect():
    summary_path=BASE/'results/stage5e_summary.json';handoff_path=BASE/'results/next_step_handoff.json'
    s=read(summary_path);h=read(handoff_path)
    report=Path(s['report']['path']);check(s['report']);text=report.read_text(encoding='utf8')
    for r in s['source_bindings']+s['independent_audits']+s['tests']['reports']:check(r)
    assert all(read(r['path'])['status']=='passed' for r in s['independent_audits'])
    assert s['protection']==read(BASE/'results/protection_summary.json')
    inventory=read(BASE/'inventory/summary.json');contexts=read(BASE/'inventory/shared_phrase_context_diversity.json')
    selection=read(BASE/'design/selection_summary.json');queue=read(BASE/'design/review_queue.json')
    assert s['selection']==selection
    assert all(inventory[k]==v for k,v in s['inventory_counts'].items())
    assert all(contexts[k]==v for k,v in s['shared_phrase_context_counts'].items())
    tests=[read(r['path']) for r in s['tests']['reports']]
    assert tests[0]['passed']==6 and tests[0]['failed']==0
    assert tests[1]['passed'] is True and tests[1]['tests_run']==16 and not tests[1]['failures'] and not tests[1]['errors']
    check(tests[1]['producer']);check(tests[1]['test_code'])
    assert tests[2]['status']=='passed' and tests[2]['tests_run']==8 and tests[2]['failures']==tests[2]['errors']==tests[2]['exit_code']==0
    for r in tests[2]['source_bindings']:check(r)
    assert s['tests']['passed_count']==30 and s['tests']['whole_project_tests'] is False
    assert s['scope']['metadata_only'] is True and all(v==0 for k,v in s['scope'].items() if k!='metadata_only')
    assert s['source_training_ready'] is False and s['source_provisional'] is True and s['prior5c_strict_fit_gate_passed'] is False
    caches=read(S3/'results/cache_handoff.json')['records']
    assert s['preserved_boundary_flags']=={'existing_thirty_cache_clips':sum(r['boundary_followup_flag'] for r in caches),
        'existing_train_phrase_subset':sum(r['boundary_followup_flag'] for r in caches if r['split']=='train' and r['kind']=='phrase')}
    assert s['next_stage']=='5f' and s['next_stage_status']=='not_executed' and not s['automatic_sentence_expansion']
    for name,path in [('summary',summary_path),('report',report),('queue',BASE/'design/review_queue.json'),
                      ('role_proposal',BASE/'design/role_proposal.json'),('protected_scope',BASE/'data/protected_scope.json')]:assert h[name]==binding(path)
    assert h['first_batch']==selection['first_batch']==queue['first_batch']
    assert h['first_batch']['review_ids']==[f'Q{i:02d}' for i in range(1,9)]
    assert h['first_batch']['speaker_count']==8 and h['next_stage']=='5f'
    assert h['reserved_probe_recordings']==selection['reserved_recording_count']==7
    assert h['reserved_eligible_parent_count']==selection['reserved_eligible_parent_count']==61
    assert h['reserved_unselected_sibling_count']==selection['reserved_unselected_sibling_count']==54
    for k in ['new_training_allowed_in_next_review_stage','new_cache_auto_promotion','training_ready','immediate_additional_download_needed','immediate_colab_needed','automatic_sentence_stage6']:assert h[k] is False
    for k in ['actual_new_clips','actual_new_human_approvals','service_changes']:assert h[k]==0
    for k in ['source_provisional','original_strict_gate_unchanged','current_dev3_and_all_historical_protection_preserved','pending_old_listening_answers_remain_pending']:assert h[k] is True
    assert h['held_acoustic_OOV_parent_still_held']=='lip_J_1_F_03_C293_A_012_019'
    # Match every displayed queue entry to the bound candidate metadata.
    qlines=[line for line in text.splitlines() if re.match(r'^\| Q\d{2} \|',line)]
    assert len(qlines)==20
    for line,row in zip(qlines,queue['records']):
        cells=[x.strip() for x in line.split('|')[1:-1]]
        assert cells[0]==row['review_id'] and row['parent_id'].endswith(cells[1]) and cells[2]==row['text']
        expected_role='학습' if row['proposed_role']=='proposed_train' else '같은 표현·다른 녹화' if row['proposed_role']=='same_expression_probe' else '새 구절:'
        assert cells[3].startswith(expected_role)
    for row in inventory['per_speaker']:
        expected=f"| {row['speaker_id']} | {row['parent_count']} | {row['recording_count']} | {row['shared_phrase_surfaces_with_other_speakers']} |"
        assert expected in text
    for row in queue['records']:
        if row['proposed_role']!='new_phrase_probe':continue
        expected_words=', '.join(row['unseen_train_eojeols']) or '없음'
        pct=lambda v:f'{v*100:.2f}'.rstrip('0').rstrip('.')+'%'
        expected=f"| {row['text']} | {expected_words} | {pct(row['eojeol_occurrence_coverage'])} | {pct(row['syllable_occurrence_coverage'])} |"
        assert expected in text,expected
    links=[]
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        assert not target.startswith('http')
        resolved=(report.parent/target).resolve()
        assert resolved.is_file() and resolved.is_relative_to(ROOT),target
        links.append(binding(resolved))
    assert len(links)==9
    for required in ['266개 / 8명 / 32녹화','5,297','5,160','65종','23종','25구절·17녹화·8명·21종','61개','54개','30개(보호 6·후보 집계 16·선정 8)','1,860개','재사용 개발 3구절','독립적인 최종 시험이 아니다','CTC 가능성은 미측정','새 어절','기존 5c','6단계 문장 확대나 추가 학습으로 바로 넘어가지 않는다']:
        assert required in text,required
    final_source=read(HERE/'final_source_preservation.json')
    assert final_source['status']=='passed' and final_source['verified_frozen_artifact_entries']==1860
    assert final_source['stage1_manifest_artifact_entries']==10 and final_source['scope']['verified_artifact_entries']==10
    for path in BASE.rglob('*'):
        if path.is_file():assert path.suffix.lower() not in {'.pt','.npz','.wav','.mp4','.tar','.zip'}
    return {'schema':'offline-stage5e-independent-final-report-audit-v1','status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'report':binding(report),'summary':binding(summary_path),'handoff':binding(handoff_path),'publisher':binding(BASE/'publish_stage.py'),
        'linked_files':links,'checks':{'summary_numbers_match_independently_audited_sources':True,'displayed20_choices_and4_coverages_match':True,
            'all9_markdown_links_resolve':True,'tests6_plus16_plus8_equal30':True,'prior1860_plus10_plus_scope10_preserved':True,
            'unconfirmed_candidates_not_promoted':True,'next_stage8parent_review_only':True,'reused_development_limits_and_prior_failed_gate_preserved':True},
        'publisher_static_review':{'read':True,'imports_only_stdlib_and_metadata_prepare_helpers':True,
            'publish_creates_separate_json_only':True,'freeze_binds_existing_artifacts_and_report':True,
            'no_training_media_or_network_invocation':True,'actual_publisher_executions_by_auditor':0},
        'independent_audit_files':[binding(HERE/n) for n in ['source_preflight.json','protection_audit.json','inventory_audit.json','shared_context_audit.json','design_audit.json','final_source_preservation.json']],
        'raw_media_reads':0,'NPZ_arrays_loaded':0,'checkpoint_tensor_loads':0,'new_neural_calls':0,'optimizer_updates':0}


if __name__=='__main__':
    path=HERE/'final_report_audit.json';assert not path.exists()
    result=inspect();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'verified_links':len(result['linked_files']),'tests':30}))
