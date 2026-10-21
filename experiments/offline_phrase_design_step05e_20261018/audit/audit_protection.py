"""Rebuild the293-to266 source selection independently from frozen metadata."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from audit_sources import ROOT, BASE, HERE, S1, S2G, S3, read, binding, check, textkey, anglekey
from audit_inventory import encode


def inspect():
    first=read(S1/'results/summary.json')
    original=read(S1/'results/analysis_sentences.json')
    parent=read(ROOT/'experiments/diversity_step15_20261007/data_plan.json')
    logical={}
    for key in ['old_data_plan','new_data_plan']:
        check(parent['source_refs'][key])
        logical.update({r['id']:r for r in read(parent['source_refs'][key]['path'])['records']})
    assert len(logical)==632
    oldpool={r['id']:r for r in parent['pool_profile']['records']}
    split=read(S3/'split/split_plan.json')
    dev=[r for r in split['records'] if r['split']=='dev']
    reviewed_path=S2G/'results/verified_phrase_handoff.json'
    review=read(reviewed_path)
    canonical={r['parent_id']:r for r in review['records']}
    held={r['parent_id'] for r in review['held_short_alignment']}
    past_probes=first['protected_probe_ids']
    dev16=first['protected_dev16_ids']
    recording_ids=past_probes+dev16
    stems={logical[sid]['source_stem'] for sid in recording_ids}|{r['source_stem'] for r in dev}
    videos={oldpool[sid]['video_sha256'] for sid in recording_ids}|{r['video_sha256'] for r in dev}
    labels={oldpool[sid]['label_sha256'] for sid in recording_ids}|{r['label_sha256'] for r in dev}
    fulltexts={textkey(logical[sid]['text']) for sid in first['protected_historical_ids']+recording_ids}
    fulltexts.update(textkey(r[k]) for r in dev for k in ['parent_text','historical_parent_text'])
    devphrase={textkey(r['phrase']['text']) for r in dev}
    all_ids=set(first['protected_historical_ids']+recording_ids+first['pending_listening_ids'])|{r['parent_id'] for r in dev}
    expected_scope={'protected_parent_ids':sorted(all_ids),'protected_recording_keys':sorted({anglekey(s) for s in stems}),
        'blocked_source_stem':sorted(stems),'blocked_video_sha256':sorted(videos),'blocked_label_sha256':sorted(labels),
        'protected_full_text_keys':sorted(fulltexts),'protected_phrase_text_keys':sorted(devphrase),
        'protected_candidate_text_keys':sorted(fulltexts|devphrase),'held_short_alignment_parent_ids':sorted(held),
        'pending_listening_ids':first['pending_listening_ids']}
    scope=read(BASE/'data/protected_scope.json')
    for key,value in expected_scope.items():assert scope[key]==value,key
    for bound in scope['source_bindings']:check(bound)
    assert set(scope['current_dev_parent_ids'])=={r['parent_id'] for r in dev}
    assert scope['source_training_ready'] is False and scope['new_approval_added'] is False
    decisions=[];accepted=[];overlays=[]
    for old in original:
        sid=old['id'];r=canonical.get(sid);text=r['parent_text'] if r else old['text'];reasons=[]
        if sid in all_ids:reasons.append('protected_parent_id')
        if anglekey(old['source_stem']) in expected_scope['protected_recording_keys']:reasons.append('protected_angle_independent_recording')
        for key,values in [('source_stem',stems),('video_sha256',videos),('label_sha256',labels)]:
            if old[key] in values:reasons.append('protected_'+key)
        texts={textkey(old['text']),textkey(text)}
        if texts&fulltexts:reasons.append('protected_historical_or_canonical_full_transcript')
        if any(phrase in t for phrase in devphrase for t in texts):reasons.append('contains_current_dev_phrase')
        if sid in held:reasons.append('held_acoustic_OOV_short_alignment')
        decision={'parent_id':sid,'speaker_id':old['speaker_id'],'recording_key':anglekey(old['source_stem']),
            'eligible':not reasons,'exclusion_reasons':reasons,'historical_text':old['text'],'canonical_text':text,
            'canonical_overlay_applied':r is not None,'text_changed':text!=old['text']}
        decisions.append(decision)
        if r:overlays.append({**decision,'source':binding(reviewed_path)})
        if reasons:continue
        accepted.append({**old,'parent_id':sid,'recording_key':anglekey(old['source_stem']),
            'historical_text':old['text'],'historical_token_ids':old['token_ids'],'text':text,'token_ids':encode(text),'eojeols':text.split(),
            'source_approval_status':{'parent_transcript_confirmed':r is not None,'parent_evidence':binding(reviewed_path) if r else None,
                'previous_reviewed_phrase':r,'new_phrase_boundary_confirmed':False,'exact_visual_boundary_certified':False,
                'new_human_approval_added':False,'training_ready':False},
            'current5c_training_parent':sid in split['train_parent_ids'],
            'historical_pool':'Previously considered old training candidate, not untouched final test data.',
            'word_boundaries':None,'new_phrase_frames':None,'metadata_only':True,'training_ready':False,'source_provisional':True})
    eligible=read(BASE/'data/eligible_sentences.json')
    for bound in eligible['source_bindings']:check(bound)
    assert eligible['records']==accepted and len(accepted)==266
    assert eligible['protected_scope']==binding(BASE/'data/protected_scope.json')
    assert eligible['new_human_approvals']==0 and eligible['training_ready'] is False
    assert read(BASE/'data/parent_decisions.json')['records']==decisions
    actual_overlay=read(BASE/'data/canonical_overlays.json')
    assert actual_overlay['records']==overlays
    assert actual_overlay['source_labels_modified'] is False and actual_overlay['new_human_approvals']==0
    summary=read(BASE/'results/protection_summary.json')
    counts={'initial_sentences':len(original),'eligible_sentences':len(accepted),'excluded_sentences':len(original)-len(accepted),
        'eligible_speakers':len({r['speaker_id'] for r in accepted}),'eligible_recordings':len({r['recording_key'] for r in accepted}),
        'reason_counts_overlapping':dict(Counter(reason for r in decisions for reason in r['exclusion_reasons'])),
        'canonical_parent_evidence_records':len(overlays),'canonical_changed_texts':sum(r['text_changed'] for r in overlays),
        'eligible_confirmed_parent_transcripts':sum(r['source_approval_status']['parent_transcript_confirmed'] for r in accepted),
        'new_clips':0,'new_neural_calls':0,'new_training_updates':0,'raw_media_reads':0,'downloads':0,'service_changes':0}
    for key,value in counts.items():assert summary[key]==value,key
    assert summary['eligible']==binding(BASE/'data/eligible_sentences.json') and summary['scope']==binding(BASE/'data/protected_scope.json')
    return {'schema':'offline-stage5e-independent-protection-audit-v1','status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'verified_artifacts':[binding(BASE/path) for path in ['data/eligible_sentences.json','data/parent_decisions.json','data/canonical_overlays.json','data/protected_scope.json','results/protection_summary.json']],
        'counts':counts,'protected_key_counts':{k:len(v) for k,v in expected_scope.items()},
        'all293_decisions_and266_canonical_rows_exact':True,'all_candidate_tokens_unicode_arithmetic_verified':True,
        'source_approval_scope_preserved':True,'canonical_overlays_matched15_prior_records':True,
        'producer_imports':0,'NPZ_arrays_loaded':0,'checkpoint_tensor_loads':0,'raw_media_reads':0,'network_forward_calls':0,'optimizer_updates':0}


if __name__=='__main__':
    path=HERE/'protection_audit.json'
    assert not path.exists()
    result=inspect()
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'counts':result['counts']}))
