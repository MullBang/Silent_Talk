"""Independently audit fixed review choices, role isolation and lexical coverage."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import json
from audit_sources import ROOT, BASE, HERE, read, check, binding, textkey


def counts(rows):
    return {'records':len(rows),'parents':len({r['parent_id'] for r in rows}),
        'recordings':len({r['recording_key'] for r in rows}),'speakers':len({r['speaker_id'] for r in rows}),
        'unique_phrase_texts':len({textkey(r['text']) for r in rows})}


def compact(row):return {k:row[k] for k in ['parent_id','speaker_id','recording_key','text']}


def substring(needle,haystack):
    a=tuple(needle.split());b=tuple(haystack.split())
    return a in {b[i:i+len(a)] for i in range(len(b)-len(a)+1)}


def inspect():
    folder=BASE/'design'
    queue=read(folder/'review_queue.json');role=read(folder/'role_proposal.json');summary=read(folder/'selection_summary.json')
    policy=read(folder/'selection_policy.json')
    for r in queue['sources']:check(r)
    assert role['sources']==summary['sources']==queue['sources']
    parents={r['parent_id']:r for r in read(BASE/'data/eligible_sentences.json')['records']}
    candidates={r['candidate_id']:r for r in read(BASE/'inventory/phrase_candidates.json')['records']}
    protected=read(BASE/'data/protected_scope.json')
    selected=queue['records'];anchors=role['existing_anchors']
    assert len(selected)==len({r['parent_id'] for r in selected})==len({r['candidate_id'] for r in selected})==20
    assert len(selected)<=policy['hard_constraints']['maximum_new_review_parents']
    expected_anchors=[]
    for old in read(ROOT/'experiments/offline_phrase_only_step05c_20261018/evaluation_plan.json')['items']:
        p=parents[old['parent_id']]
        expected_anchors.append({'parent_id':old['parent_id'],'clip_id':old['clip_id'],'speaker_id':p['speaker_id'],
            'recording_key':p['recording_key'],'source_stem':p['source_stem'],'text':old['text'],
            'proposed_role':'existing_train_anchor','source_approval_status':p['source_approval_status'],
            'new_review_required_here':False,'new_human_approval_added':False,'training_ready':False})
        assert old['split']=='train' and old['kind']=='phrase'
    assert anchors==expected_anchors and len(anchors)==12
    tree=ast.parse((folder/'select_review_queue.py').read_text(encoding='utf8'))
    choices=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHOICES' for t in n.targets))
    for i,(row,choice) in enumerate(zip(selected,choices),1):
        suffix,text,proposed_role,rationale=choice
        assert row['parent_id'].endswith(suffix) and row['text']==text and row['proposed_role']==proposed_role and row['selection_rationale']==rationale
        assert row['review_order']==i and row['review_id']==f'Q{i:02d}'
        actual=candidates[row['candidate_id']];parent=parents[row['parent_id']]
        assert all(row[k]==v for k,v in actual.items())
        assert not parent['current5c_training_parent']
        assert row['parent_text']==parent['text'] and row['parent_historical_text']==parent['historical_text'] and row['parent_source_text']==parent['source_text']
        assert row['parent_transcript_confirmed']==parent['source_approval_status']['parent_transcript_confirmed']==False
        assert row['recording_video_sha256']==parent['video_sha256'] and row['recording_label_sha256']==parent['label_sha256']
        assert row['reference_parent_frames_not_phrase_frames']==parent['frames']
        assert row['actual_split'] is None and row['n_eojeols'] in (2,3,4)
        assert row['frame_count'] is None and row['ctc_feasible'] is None and not row['candidate_boundary_confirmed']
        assert not row['training_ready'] and not row['new_human_approval_added']
        assert row['parent_id'] not in protected['protected_parent_ids']+protected['held_short_alignment_parent_ids']
        assert row['recording_key'] not in protected['protected_recording_keys']
        assert parent['video_sha256'] not in protected['blocked_video_sha256'] and parent['label_sha256'] not in protected['blocked_label_sha256']
        assert textkey(row['text']) not in protected['protected_candidate_text_keys']
        assert not any(p in textkey(row['text']) for p in protected['protected_phrase_text_keys'])
        assert not row['contained_protected_full_text_keys_without_exact_match']
    assert len(choices)==20
    new_train=[r for r in selected if r['proposed_role']=='proposed_train']
    probes=[r for r in selected if r['proposed_role']!='proposed_train']
    train=anchors+new_train
    assert role['proposed_train_candidate_ids']==[r['candidate_id'] for r in new_train]
    assert role['probe_records']==probes
    assert len(train)==25 and len(probes)==7
    overlap_counts={}
    for field in ['parent_id','recording_key']:
        overlap_counts[field]=len({r[field] for r in train}&{r[field] for r in probes})
    for field in ['video_sha256','label_sha256']:
        overlap_counts[field]=len({parents[r['parent_id']][field] for r in train}&{parents[r['parent_id']][field] for r in probes})
    overlap_counts['old_or_canonical_parent_text']=len({textkey(parents[r['parent_id']][k]) for r in train for k in ['text','historical_text']} & {textkey(parents[r['parent_id']][k]) for r in probes for k in ['text','historical_text']})
    assert not any(overlap_counts.values())
    train_words={w for r in train for w in r['text'].split()}
    train_syllables={c for r in train for c in r['text'] if 0xAC00<=ord(c)<=0xD7A3}
    overlap=[]
    for row in probes:
        exact=[compact(r) for r in train if textkey(r['text'])==textkey(row['text'])]
        chars=[compact(r) for r in train if textkey(row['text']) in textkey(r['text'])]
        eojeols=[compact(r) for r in train if substring(row['text'],r['text'])]
        words=row['text'].split();syllables=[c for c in row['text'] if 0xAC00<=ord(c)<=0xD7A3]
        seen=[w for w in words if w in train_words];unseen=[w for w in words if w not in train_words]
        subtype='same_exact_expression' if exact else 'new_phrase_with_unseen_eojeols' if unseen else 'all_eojeols_seen_new_order_or_combination'
        assert row['observed_train_eojeols']==seen and row['unseen_train_eojeols']==unseen
        assert row['eojeol_occurrence_coverage']==len(seen)/len(words)
        assert row['syllable_occurrence_coverage']==sum(c in train_syllables for c in syllables)/len(syllables)
        assert row['probe_subtype']==subtype
        if row['proposed_role']=='new_phrase_probe':assert not exact and not chars and not eojeols
        else:assert row['proposed_role']=='same_expression_probe' and exact
        overlap.append({'review_id':row['review_id'],'text':row['text'],'proposed_role':row['proposed_role'],
            'exact_train_matches':exact,'probe_contained_in_train_chars':chars,'probe_contained_in_train_eojeols':eojeols,
            'eojeol_occurrence_coverage':len(seen)/len(words),'unseen_eojeols':unseen,'subtype':subtype})
    assert role['overlap_analysis']==overlap
    reservations=role['reserved_recordings']
    assert len(reservations)==len({r['recording_key'] for r in probes})==7
    for got in reservations:
        recording=got['recording_key'];siblings=[r for r in parents.values() if r['recording_key']==recording]
        chosen=[r for r in probes if r['recording_key']==recording];assert len(chosen)==1
        exp={'recording_key':recording,'proposed_role':chosen[0]['proposed_role'],
            'selected_parent_ids':[r['parent_id'] for r in chosen],
            'reserved_parent_ids_in_current_eligible_pool':sorted(r['parent_id'] for r in siblings),
            'unselected_siblings_reserved_from_future_training':sorted(r['parent_id'] for r in siblings if r['parent_id']!=chosen[0]['parent_id']),
            'blocked_source_stems':sorted({r['source_stem'] for r in siblings}),
            'blocked_video_sha256':sorted({r['video_sha256'] for r in siblings}),
            'blocked_label_sha256':sorted({r['label_sha256'] for r in siblings}),
            'applies_to_all_angles_and_future_discovered_siblings':True,'proposal_only_not_actual_split_assignment':True}
        assert got==exp
    by_speaker=[]
    for speaker in sorted({r['speaker_id'] for r in train}):
        rows=[r for r in train if r['speaker_id']==speaker]
        by_speaker.append({'speaker_id':speaker,'existing_anchors':sum(r['proposed_role']=='existing_train_anchor' for r in rows),
            'new_train_proposals':sum(r['proposed_role']=='proposed_train' for r in rows),
            'unique_proposed_train_phrases':len({textkey(r['text']) for r in rows}),'phrases':[r['text'] for r in rows]})
    assert summary['speaker_train_diversity']==role['speaker_train_diversity']==by_speaker
    assert all(r['unique_proposed_train_phrases']>=3 for r in by_speaker)
    groups=defaultdict(list)
    for r in train:groups[textkey(r['text'])].append(r)
    shared=[]
    for rows in groups.values():
        speakers=sorted({r['speaker_id'] for r in rows})
        if len(speakers)<2:continue
        shared.append({'text':rows[0]['text'],'speakers':speakers,'parent_count':len({r['parent_id'] for r in rows}),
            'recording_count':len({r['recording_key'] for r in rows}),'full_parent_text_count':len({textkey(parents[r['parent_id']]['text']) for r in rows})})
    assert summary['shared_training_expressions']==role['shared_training_expressions']==shared
    for field,rows in [('queue_counts',selected),('existing_anchor_counts',anchors),('combined_proposed_train_counts',train),('probe_counts',probes)]:assert summary[field]==counts(rows)
    expected_summary={'queue_role_counts':dict(Counter(r['proposed_role'] for r in selected)),
        'new_phrase_subtypes':dict(Counter(r['probe_subtype'] for r in probes if r['proposed_role']=='new_phrase_probe')),
        'reserved_recording_count':7,'reserved_eligible_parent_count':sum(len(r['reserved_parent_ids_in_current_eligible_pool']) for r in reservations),
        'reserved_unselected_sibling_count':sum(len(r['unselected_siblings_reserved_from_future_training']) for r in reservations),
        'review_eojeol_lengths':dict(Counter(str(r['n_eojeols']) for r in selected)),
        'exact_ending_only_selected_count':sum(r['text'] in ['것 같아','수 있어','할 수','있는 것'] for r in selected),
        'all_new_queue_parent_transcripts_unconfirmed':True,'all_new_phrase_boundaries_unconfirmed':True}
    for field,value in expected_summary.items():assert summary[field]==value,field
    assert summary['reserved_eligible_parent_count']==61 and summary['reserved_unselected_sibling_count']==54
    assert queue['first_batch']==summary['first_batch']
    first=queue['first_batch']
    for field,key in [('review_ids','review_id'),('parent_ids','parent_id'),('candidate_ids','candidate_id')]:assert first[field]==[r[key] for r in selected[:8]]
    assert first['speaker_count']==len({r['speaker_id'] for r in selected[:8]})==8
    assert all(r['proposed_role']=='proposed_train' for r in selected[:8])
    assert not queue['audio_clip_generation'] and queue['new_boundary_reviews']==0 and queue['source_approval_promotions']==0 and not queue['training_ready']
    assert role['actual_split_assignment'] is None and role['source_training_ready'] is False and role['no_independent_final_test_claim']
    for field in ['new_media_reads','new_downloads','new_neural_calls','new_optimizer_updates','new_human_approvals','new_clips','new_splits']:assert summary[field]==0
    assert not list(BASE.rglob('*.pt')) and not list(BASE.rglob('*.npz')) and not list(BASE.rglob('*.wav')) and not list(BASE.rglob('*.mp4'))
    return {'schema':'offline-stage5e-independent-design-audit-v1','status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'artifacts':[binding(folder/n) for n in ['selection_policy.json','review_queue.json','role_proposal.json','selection_summary.json','select_review_queue.py']],
        'counts':{k:summary[k] for k in ['queue_counts','existing_anchor_counts','combined_proposed_train_counts','probe_counts','queue_role_counts','new_phrase_subtypes','reserved_eligible_parent_count','reserved_unselected_sibling_count']},
        'independent_overlap_counts':overlap_counts,'all7_probe_exact_containment_and_coverage_checks_pass':True,
        'first_batch_eight_speakers_training_proposals_only':True,'whole_recording_reservations_exact':True,
        'new_media_reads':0,'new_neural_calls':0,'optimizer_updates':0,'NPZ_arrays_loaded':0,'checkpoint_tensor_loads':0,
        'producer_imports':0,'actual_split_assignment':None}


if __name__=='__main__':
    path=HERE/'design_audit.json';assert not path.exists()
    result=inspect();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'counts':result['counts']}))
