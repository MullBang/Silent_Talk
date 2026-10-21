"""Preserve raw labels and hold unresolved flat alternative transcriptions.

This module never edits a source, runs a model, or approves a transcript for
training. Ordinary text reuses the hash-pinned existing tokenizer unchanged.
Only explicit, individually bound human evidence can select an alternative;
an ASR output is never a selection rule. Resolving a notation choice does not
certify the complete utterance, its boundaries, or its suitability for training.

Public APIs:
  parse_source_text(raw_text): pure syntax parser, no file access.
  resolve_record(record, override=None): preserve one selection row's targets.
  build_targets(records, overrides=()): preserve the frozen120 order and roles.

Override shape:
  {id, raw_text_sha256, choices:[{alternative_id:'alt_0', option:0|1}],
   evidence:{path:<workspace JSON>, sha256:<file SHA>}}
The evidence document repeats id/raw_text_sha256/choices and has schema
lipnet-step07-alternative-resolution-evidence-v1, evidence_type user_listening
or documented_individual_override, a nonempty selection_statement,
scope alternative_choice_only, and full_transcript_human_approved false.
The caller is responsible for recording authentic human evidence; checking
this schema and its hash cannot independently prove who listened.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import re
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
TOKENIZER=ROOT/'backend/models/tokenizer.py'
TOKENIZER_SHA='c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714'
POLICY_VERSION='ko-alternative-target-v1'
SCHEMA='lipnet-transcript-step07-targets-v1'
MAX_ALTERNATIVES=4
PAIR=re.compile(r'\(([^()]+)\)/\(([^()]+)\)')
SLASHES=set('/\\\u2044\u2215\uff0f\uff3c')
_tokenizer=None
SOURCE_PLAN=ROOT/'experiments/new_data_step06_20261007/source_plan.json'
SOURCE_PLAN_SHA='5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929'
MEDIA_MANIFEST=ROOT/'experiments/new_data_step06_20261007/results/media_manifest.json'
MEDIA_MANIFEST_SHA='fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8'
OVERRIDES=BASE/'label_overrides.json'
OVERRIDES_SHA='cc163cc9e5ce82ef0cc807cc5cec344835e182b04b571547ecc494dc2e3fc487'
LABEL_PLAN=BASE/'label_plan.json'
OUTPUT=BASE/'results/label_targets.json'
RULES={'syntax':'strict flat (A)/(B); unknown/malformed/nested/empty syntax held',
       'candidate_limit':16,'ordinary_normalization':'unchanged hash-pinned ko-text-v1/ko-jamo-v2',
       'choice_rule':'individually bound actual human evidence; no ASR/model choice',
       'ctc_rule':'tokens plus adjacent repeats <= existing student frames; never shorten a target to fit',
       'approval_rule':'alternative choice alone adds zero complete transcript or boundary approvals',
       'source_mutation':False,'training':False,'model_execution':False}


class PolicyError(ValueError):
    """A frozen input or alleged human evidence is inconsistent."""


def require(value,message):
    if not value:
        raise PolicyError(message)


def raw_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def load_tokenizer():
    global _tokenizer
    require(file_sha256(TOKENIZER)==TOKENIZER_SHA,'Frozen tokenizer changed')
    if _tokenizer is None:
        spec=importlib.util.spec_from_file_location('step07_unchanged_ko_tokenizer',TOKENIZER)
        value=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        _tokenizer=value
    return _tokenizer


def structural(character):
    return character in SLASHES or unicodedata.category(character) in ('Ps','Pe')


def parse_source_text(raw_text):
    """Parse exact flat '(A)/(B)' only; unsupported syntax remains an explicit hold.

    Offsets are Python Unicode character indices in the untouched raw string.
    Text and alternative spans reconstruct the raw string without loss. All
    bracket families and slash forms are reserved to avoid punctuation cleanup
    silently converting an unknown annotation into ordinary speech.
    """
    if not isinstance(raw_text,str):
        raise TypeError('Source text must be a string')
    segments=[]; alternatives=[]; cursor=0; plain_start=0
    while cursor<len(raw_text):
        ch=raw_text[cursor]
        if not structural(ch):
            cursor+=1;continue
        matched=PAIR.match(raw_text,cursor) if ch=='(' else None
        if matched is None:
            return {'status':'unsupported_syntax','raw_text':raw_text,'segments':segments,
                    'alternatives':alternatives,'error':{'code':'unsupported_or_malformed_annotation',
                        'offset':cursor,'character':ch,
                        'message':'Only exact flat (A)/(B) notation is supported; do not delete brackets or slashes'}}
        options=[matched.group(1),matched.group(2)]
        if any(not option.strip() or any(structural(c) for c in option) for option in options):
            return {'status':'unsupported_syntax','raw_text':raw_text,'segments':segments,
                    'alternatives':alternatives,'error':{'code':'empty_or_nested_alternative',
                        'offset':cursor,'character':ch,
                        'message':'Both alternatives must be nonempty and contain no bracket/slash syntax'}}
        if plain_start<cursor:
            segments.append({'kind':'text','span':[plain_start,cursor],'raw':raw_text[plain_start:cursor]})
        alternative={'alternative_id':f'alt_{len(alternatives)}','span':[cursor,matched.end()],
                     'raw':matched.group(0),'options':options,
                     'option_spans':[[matched.start(1),matched.end(1)],[matched.start(2),matched.end(2)]]}
        alternatives.append(alternative)
        segments.append({'kind':'alternative',**deepcopy(alternative)})
        cursor=matched.end();plain_start=cursor
    if plain_start<len(raw_text):
        segments.append({'kind':'text','span':[plain_start,len(raw_text)],'raw':raw_text[plain_start:]})
    require(''.join(segment['raw'] for segment in segments)==raw_text,'Parser did not preserve raw text')
    return {'status':'alternatives' if alternatives else 'plain','raw_text':raw_text,
            'segments':segments,'alternatives':alternatives,'error':None}


def choice_list(parsed,values):
    return [{'alternative_id':a['alternative_id'],'option':option}
            for a,option in zip(parsed['alternatives'],values)]


def render_choice(parsed,choices):
    """Render a complete explicit choice without applying text normalization."""
    require(parsed['status'] in ('plain','alternatives'),'Cannot render unsupported syntax')
    ids=[a['alternative_id'] for a in parsed['alternatives']]
    require(isinstance(choices,list) and len(choices)==len(ids),'Need one choice per alternative')
    require([c.get('alternative_id') for c in choices]==ids,'Alternative choices missing, duplicated, or reordered')
    require(all(type(c.get('option')) is int and c['option'] in (0,1) for c in choices),'Alternative option must be integer0 or1')
    selected={c['alternative_id']:c['option'] for c in choices}
    return ''.join(s['raw'] if s['kind']=='text' else s['options'][selected[s['alternative_id']]]
                   for s in parsed['segments'])


def target_info(text,student_frames,tokenizer):
    normalized=tokenizer.normalize_text(text)
    tokens=tokenizer.encode_text(text)
    minimum=tokenizer.ctc_min_input_len(tokens)
    require(normalized and tokens,'Empty normalized target is not a valid transcription')
    return {'normalized_text':normalized,'tokens':tokens,
            'tokens_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode('ascii')).hexdigest(),
            'ctc_min_frames':minimum,'student_frames':student_frames,'ctc_feasible':minimum<=student_frames}


def validate_override(override,record,parsed):
    """Validate an individual human evidence file without treating it as full approval."""
    require(isinstance(override,dict),'Override must be an object')
    require(parsed['status']=='alternatives','An override cannot replace ordinary or malformed text')
    require(override.get('id')==record['id'] and override.get('raw_text_sha256')==raw_sha256(record['source_text']),
            'Override ID/raw text binding differs')
    choices=override.get('choices')
    render_choice(parsed,choices)
    ref=override.get('evidence')
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'Evidence needs an exact path/SHA binding')
    path=Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT),'Choice evidence must be a local workspace file')
    require(file_sha256(path)==ref['sha256'],'Human choice evidence file changed')
    evidence=json.loads(path.read_text(encoding='utf-8-sig'))
    require(evidence.get('schema')=='lipnet-step07-alternative-resolution-evidence-v1','Unsupported selection evidence schema')
    require(evidence.get('id')==record['id'] and evidence.get('raw_text_sha256')==override['raw_text_sha256']
            and evidence.get('choices')==choices,'Evidence does not support this individual alternative selection')
    require(evidence.get('evidence_type') in ('user_listening','documented_individual_override'),
            'An ASR/model output is not valid choice evidence')
    require(isinstance(evidence.get('selection_statement'),str) and evidence['selection_statement'].strip(),
            'Actual user answer or individual override basis is required')
    require(evidence.get('scope')=='alternative_choice_only' and evidence.get('full_transcript_human_approved') is False,
            'Alternative selection evidence must not imply full utterance approval')
    return {'binding':{'path':str(path),'sha256':ref['sha256']},'evidence':evidence}


def resolve_record(record,override=None):
    """Derive target candidates, retaining unresolved rows rather than deleting them."""
    tokenizer=load_tokenizer()
    raw=record['source_text'];parsed=parse_source_text(raw)
    frames=record['expected_frames']
    require(type(frames) is int and frames>0,'Positive integer student frame count required')
    result={'id':record['id'],'split':record['split'],'speaker_id':record['speaker_id'],
            'source_stem':record['id'].rsplit('_',1)[0],
            'source_text':raw,'raw_text_sha256':raw_sha256(raw),'original_normalized_text':record['text'],
            'start_time':record.get('start_time'),'end_time':record.get('end_time'),
            'label_sha256':record.get('label_sha256'),'student_frames':frames,
            'policy_version':POLICY_VERSION,'tokenizer':tokenizer.tokenizer_metadata(),
            'parser':parsed,'previous_target':None,'candidates':[],'selected_candidate_id':None,
            'selection_evidence':None,'canonical_target':None,'canonical_tokens':None,
            'ctc_min_frames':None,'ctc_feasible':None,'ordinary_tokens_identical':None,
            'target_status':None,'hold_reason':None,'training_ready':False,
            'full_transcript_human_approved':False,'human_boundary_approved':False,
            'asr_used_to_choose_target':False}
    # This is historical provenance only: malformed notation is never admitted
    # merely because the legacy punctuation normalizer can encode it.
    try:
        previous=target_info(raw,frames,tokenizer)
    except ValueError as exc:
        result['previous_target_error']=str(exc)
        previous=None
    if previous is not None:
        require(previous['normalized_text']==record['text'],'Frozen previous normalized text differs')
        require(previous['ctc_min_frames']==record['ctc_min_frames'],'Frozen previous CTC minimum differs')
        result['previous_target']=previous
    if parsed['status']=='unsupported_syntax':
        require(override is None,'Unsupported annotation cannot be resolved by an unstructured override')
        result.update(target_status='hold_unsupported_syntax',hold_reason=parsed['error']['code'])
        return result
    if len(parsed['alternatives'])>MAX_ALTERNATIVES:
        require(override is None,'Too many alternatives require a separately reviewed policy')
        result.update(target_status='hold_candidate_limit',hold_reason='More than four alternatives; no partial Cartesian product')
        return result
    try:
        for alternative in parsed['alternatives']:
            for option in alternative['options']:
                require(tokenizer.normalize_text(option) and tokenizer.encode_text(option),
                        'Empty normalized alternative is not a speech choice')
        combinations=product((0,1),repeat=len(parsed['alternatives']))
        for index,values in enumerate(combinations):
            choices=choice_list(parsed,values)
            rendered=render_choice(parsed,choices)
            candidate={'candidate_id':f'candidate_{index}','choices':choices,'raw_candidate_text':rendered,
                       **target_info(rendered,frames,tokenizer)}
            result['candidates'].append(candidate)
    except ValueError as exc:
        result.update(target_status='hold_unsupported_target',hold_reason=str(exc))
        return result
    if parsed['status']=='plain':
        require(override is None,'Ordinary-text replacement requires a separate explicit policy')
        require(previous is not None,'A plain target must preserve its existing tokenizer contract')
        selected=result['candidates'][0]
        require(selected['tokens']==previous['tokens'],'Ordinary sentence tokens changed')
        result['ordinary_tokens_identical']=True
        status='unchanged_ordinary_target'
    elif override is None:
        result.update(target_status='needs_alternative_resolution',hold_reason='No individually verified choice evidence; both alternatives remain candidates')
        return result
    else:
        result['selection_evidence']=validate_override(override,record,parsed)
        selected=next(c for c in result['candidates'] if c['choices']==override['choices'])
        status='resolved_alternative_choice'
    result['selected_candidate_id']=selected['candidate_id']
    result['ctc_min_frames']=selected['ctc_min_frames'];result['ctc_feasible']=selected['ctc_feasible']
    if not selected['ctc_feasible']:
        result.update(target_status='hold_ctc_infeasible',hold_reason='Selected target needs more frames than available; do not shorten the target')
        return result
    result.update(target_status=status,canonical_target=selected['normalized_text'],canonical_tokens=selected['tokens'])
    return result


def build_targets(records,overrides=()):
    """Build a policy artifact for the complete unchanged new120 selection.

    This function does not write files. Its caller must bind the frozen input
    source plan and this module's code before publishing a downstream plan.
    """
    rows=list(records);overrides=list(overrides)
    ids=[r['id'] for r in rows]
    require(len(ids)==len(set(ids))==120,'Require the full120 unique IDs without omission')
    require(Counter(r['split'] for r in rows)=={'train':96,'val':24},'Preserve train96/val24 readiness split')
    train_speakers={r['speaker_id'] for r in rows if r['split']=='train'}
    val_speakers={r['speaker_id'] for r in rows if r['split']=='val'}
    require(len(train_speakers)==12 and len(val_speakers)==3 and not train_speakers.intersection(val_speakers),
            'Preserve12 train/3 development speakers with no overlap')
    override_map={o['id']:o for o in overrides}
    require(len(override_map)==len(overrides) and set(override_map).issubset(ids),'Duplicate or unknown override IDs')
    output=[resolve_record(row,override_map.get(row['id'])) for row in rows]
    require([r['id'] for r in output]==ids,'Output ID order changed')
    return {'schema':SCHEMA,'policy_version':POLICY_VERSION,'records':output,'ids':ids,
            'counts':{'records':120,'train':96,'val':24,'speakers':15,
                      'ordinary_tokens_identical':sum(r['ordinary_tokens_identical'] is True for r in output),
                      'canonical_targets_available':sum(r['canonical_target'] is not None for r in output),
                      'target_status':dict(Counter(r['target_status'] for r in output))},
            'tokenizer':{'path':str(TOKENIZER.resolve()),'sha256':TOKENIZER_SHA,**load_tokenizer().tokenizer_metadata()},
            'source_labels_modified':False,'asr_used_to_choose_targets':False,'training_ready':False,
            'full_transcript_human_approvals_added':0,
            'limits':['Syntax resolution is separate from full transcript and boundary review.',
                      'The existing normalizer and tokens are unchanged for ordinary text.',
                      'No candidate is selected by an ASR comparison, model probability or heuristic.',
                      'Evidence file binding authenticates bytes and scope; the caller must supply actual human provenance.']}


def local(path):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT),'Only workspace evidence may be read/written')
    return path


def binding(path):
    path=local(path)
    return {'path':str(path),'sha256':file_sha256(path)}


def read_json(path):
    return json.loads(local(path).read_text(encoding='utf-8-sig'))


def save_new(path,value):
    path=local(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def validate_evidence_provenance(record,override,source,media):
    """CLI-level binding to the actual frozen cohort and reviewed original WAV."""
    validated=validate_override(override,record,parse_source_text(record['source_text']))
    evidence=validated['evidence']
    require(evidence.get('source_plan')==binding(SOURCE_PLAN) and evidence.get('media_manifest')==binding(MEDIA_MANIFEST),
            'Choice evidence belongs to a different source/media artifact')
    require(evidence.get('raw_source_text')==record['source_text'],'Choice evidence raw transcript differs')
    require(evidence.get('training_ready') is False and evidence.get('original_labels_modified') is False,
            'Choice evidence cannot admit training or edit original labels')
    clip=next(c for c in media['clips'] if c['id']==record['id'])
    if evidence['evidence_type']=='user_listening':
        audio=evidence.get('audio')
        require(audio in [clip['audio'],clip['context_audio']],'Human listening evidence is not this selected original-WAV interval')
        require(binding(audio['path'])=={k:audio[k] for k in ('path','sha256')},'Bound listening audio changed')
    else:
        document=evidence.get('supporting_document')
        require(isinstance(document,dict) and set(document)=={'path','sha256'} and binding(document['path'])==document,
                'An individual documented override requires its supporting document hash')
    parsed=parse_source_text(record['source_text'])
    if len(parsed['alternatives'])==1 and 'selected_alternative_text' in evidence:
        require(evidence['selected_alternative_text']==parsed['alternatives'][0]['options'][override['choices'][0]['option']],
                'Evidence description and selected branch differ')
    return validated['binding']


def actual_inputs():
    require(file_sha256(SOURCE_PLAN)==SOURCE_PLAN_SHA,'Frozen Step6 source plan changed')
    require(file_sha256(MEDIA_MANIFEST)==MEDIA_MANIFEST_SHA,'Frozen Step6 media manifest changed')
    require(file_sha256(OVERRIDES)==OVERRIDES_SHA,'Recorded user overrides changed')
    source=read_json(SOURCE_PLAN);media=read_json(MEDIA_MANIFEST);overrides=read_json(OVERRIDES)
    ids=[r['id'] for r in source['records']]
    require(ids==[c['id'] for c in media['clips']] and len(ids)==len(set(ids))==120,'Source/media cohort order differs')
    require(len(overrides)==1 and overrides[0]['id']=='lip_J_1_F_05_C057_A_005_005','Unexpected per-user choice scope')
    for row,clip in zip(source['records'],media['clips']):
        require(row['source_text']==clip['reference'] and row['text']==clip['text'] and row['split']==clip['split']
                and row['speaker_id']==clip['speaker_id'] and row['expected_frames']==clip['timing']['student_frames'],
                'Frozen source/media label, role, or frame count differs')
    evidence=[]
    for override in overrides:
        record=next(r for r in source['records'] if r['id']==override['id'])
        evidence.append(validate_evidence_provenance(record,override,source,media))
    return source,media,overrides,evidence


def freeze():
    source,_,overrides,evidence=actual_inputs()
    load_tokenizer()
    require(not OUTPUT.exists(),'Target output already exists before plan freeze')
    plan={'schema':'lipnet-transcript-step07-label-plan-v1','created_utc':datetime.now(timezone.utc).isoformat(),
          'frozen_before_target_artifact_generation':True,'policy_version':POLICY_VERSION,
          'source_plan':binding(SOURCE_PLAN),'media_manifest':binding(MEDIA_MANIFEST),
          'overrides':binding(OVERRIDES),'choice_evidence':evidence,
          'code':[binding(Path(__file__)),binding(BASE/'test_label_policy.py'),binding(TOKENIZER)],
          'ids':[r['id'] for r in source['records']],'rules':RULES,
          'expected_transition':{'unchanged_ordinary_records':119,'individually_resolved_alternatives':1,
                                 'changed_target_id':overrides[0]['id'],
                                 'full_transcript_approvals_added':0,'training_ready':False},
          'scope':{'original_labels_modified':False,'source_media_read':False,'listening_audio_hash_verified':True,
                   'new_asr_inference':False,'student_training':False,'final_test_used':False}}
    save_new(LABEL_PLAN,plan)
    print(json.dumps({'label_plan':binding(LABEL_PLAN)},ensure_ascii=False),flush=True)


def verify_plan(expected_sha256):
    require(isinstance(expected_sha256,str) and file_sha256(LABEL_PLAN)==expected_sha256,'Explicit frozen label-plan SHA must match')
    plan=read_json(LABEL_PLAN)
    require(plan['schema']=='lipnet-transcript-step07-label-plan-v1' and plan['rules']==RULES,'Label rules/schema differ')
    for ref in [plan['source_plan'],plan['media_manifest'],plan['overrides'],*plan['choice_evidence'],*plan['code']]:
        require(binding(ref['path'])==ref,'Bound label execution evidence changed: '+ref['path'])
    source,media,overrides,evidence=actual_inputs()
    require(plan['ids']==[r['id'] for r in source['records']] and plan['choice_evidence']==evidence,'Frozen label order/evidence differs')
    return plan,source,media,overrides


def run(expected_sha256):
    plan,source,_,overrides=verify_plan(expected_sha256)
    artifact=build_targets(source['records'],overrides)
    changes=[{'id':r['id'],'previous_target':r['previous_target'],'canonical_target':r['canonical_target'],
              'canonical_tokens':r['canonical_tokens'],'ctc_min_frames':r['ctc_min_frames'],
              'student_frames':r['student_frames'],'selection_evidence':r['selection_evidence']}
             for r in artifact['records'] if r['canonical_tokens']!=r['previous_target']['tokens']]
    require(artifact['counts']['ordinary_tokens_identical']==119 and len(changes)==1
            and changes[0]['id']==plan['expected_transition']['changed_target_id'],'Unexpected target transition outside the single evidenced choice')
    require(artifact['counts']['target_status']=={'unchanged_ordinary_target':119,'resolved_alternative_choice':1}
            and all(r['ctc_feasible'] for r in artifact['records']),'Unexpected hold or CTC failure; do not publish a ready-looking target artifact')
    artifact.update(label_plan=binding(LABEL_PLAN),source_plan=plan['source_plan'],media_manifest=plan['media_manifest'],
                    overrides=plan['overrides'],target_changes=changes,
                    created_utc=datetime.now(timezone.utc).isoformat())
    verify_plan(expected_sha256)
    save_new(OUTPUT,artifact)
    print(json.dumps({'label_targets':binding(OUTPUT),'counts':artifact['counts'],
                      'changed_target':changes[0]['canonical_target'],'old_ctc_min':changes[0]['previous_target']['ctc_min_frames'],
                      'new_ctc_min':changes[0]['ctc_min_frames']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','verify-plan'])
    parser.add_argument('--expected-plan-sha256')
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.expected_plan_sha256)
    else:
        verify_plan(args.expected_plan_sha256)
        print(json.dumps({'label_plan':binding(LABEL_PLAN),'verified':True}),flush=True)


if __name__=='__main__':
    main()
