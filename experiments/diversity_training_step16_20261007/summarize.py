"""Report the fixed final checkpoint and retain every predeclared evaluation row."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import unicodedata

BASE = Path(__file__).resolve().parent
SEEDS, COHORTS = (42,43), ('train24','train72')
UPDATES = (1200,2400,3000,3600)


def require(value,message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path=Path(path).resolve()
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest}


def aggregate(rows):
    require(rows and len({r['id'] for r in rows})==len(rows),'Missing/duplicate rows')
    counts={key:sum(row['metrics'][key] for row in rows) for key in ('character_edits','reference_characters','word_edits','reference_words')}
    require(counts['reference_characters']>0 and counts['reference_words']>0,'Empty denominator')
    return {**counts,'clips':len(rows),'failed_inputs':sum(row['status']!='ok' for row in rows),
            'cer_percent':100*counts['character_edits']/counts['reference_characters'],
            'wer_percent':100*counts['word_edits']/counts['reference_words'],
            'exact_sentences':sum(row['metrics']['exact'] for row in rows)}


def gate(deltas,failed):
    require(len(deltas)==2,'Both fixed seeds required')
    mean=statistics.mean(deltas)
    return {'seed_deltas_cer_percentage_points':deltas,'mean_delta_cer_percentage_points':mean,
            'both_seeds_improved':all(delta<0 for delta in deltas),
            'mean_improvement_at_least_5_percentage_points':mean<=-5.,'failed_inferences':failed,
            'passed':failed==0 and all(delta<0 for delta in deltas) and mean<=-5.,
            'statistical_significance_established':False,'webcam_readiness_established':False}


def validate_evaluation(value,design):
    require(value['schema']=='lipnet-step16-diversity-evaluation-v1','Wrong evaluation schema')
    seed,cohort=value['seed'],value['cohort']
    require(seed in SEEDS and cohort in COHORTS,'Unknown comparison run')
    train=design['cohorts'][cohort];dev=design['cohorts']['dev16'];probe=design['evaluation']['historical_probe14_ids']
    require([m['completed_updates'] for m in value['milestones']]==list(UPDATES),'Milestone selection changed')
    count=0
    for m in value['milestones']:
        update=m['completed_updates']
        require(m['model_state_unchanged'] and m['model_state_before']==m['model_state_after'],'Evaluation changed model')
        require(set(m['conditions'])==({'normal','static','shuffled'} if update==3600 else {'normal'}),'Missing condition')
        for condition,content in m['conditions'].items():
            ids=(train if update in (2400,3600) else [])+dev+(probe if update==3600 else []) if condition=='normal' else dev
            rows=content['records']
            require([row['id'] for row in rows]==ids,'Missing, selected or reordered evaluation rows')
            require(all(row['status'] in ('ok','failed') and (row['status']=='ok' or row['prediction']=='') for row in rows),
                    'Failures must remain explicit empty predictions')
            metric=aggregate([row for row in rows if row['id'] in dev])
            saved=content['groups']['dev16']['aggregate']
            require(all(saved[key]==v for key,v in metric.items()),'Reported dev metric differs from fixed rows')
            count+=len(rows)
    require(count==value['record_count']==2*len(train)+110,'Evaluation denominator differs from design')
    return count


def make_comparison(evaluations,design):
    lookup={(value['seed'],value['cohort']):value for value in evaluations}
    require(len(evaluations)==len(lookup)==4 and set(lookup)=={(s,c) for s in SEEDS for c in COHORTS},'Incomplete paired runs')
    models=[];failures=0;total=0
    for seed in SEEDS:
        for cohort in COHORTS:
            value=lookup[seed,cohort];total+=validate_evaluation(value,design)
            failures+=sum(row['status']!='ok' for m in value['milestones'] for c in m['conditions'].values() for row in c['records'])
            final=value['milestones'][-1]
            models.append({'seed':seed,'cohort':cohort,'final_update':3600,
                           'final_normal':final['conditions']['normal']['groups'],
                           'final_dev_conditions':{c:final['conditions'][c]['groups']['dev16'] for c in ('normal','static','shuffled')},
                           'final_normal_per_speaker':final['conditions']['normal']['per_speaker'],
                           'dev_learning_curve':[{'update':m['completed_updates'],'metrics':m['conditions']['normal']['groups']['dev16']}
                                                 for m in value['milestones']],
                           'training_learning_curve':[{'update':m['completed_updates'],
                                                       'train_native':m['conditions']['normal']['groups']['train_native'],
                                                       'train_common24':m['conditions']['normal']['groups']['train_common24']}
                                                      for m in value['milestones'] if m['completed_updates'] in (2400,3600)]})
    require(total==824,'Overall evaluation denominator changed')
    pairs=[]
    for seed in SEEDS:
        small=next(m for m in models if m['seed']==seed and m['cohort']=='train24')
        large=next(m for m in models if m['seed']==seed and m['cohort']=='train72')
        a,b=(m['final_normal']['dev16']['aggregate'] for m in (small,large))
        require(a['reference_characters']==b['reference_characters'] and a['clips']==b['clips']==16,'Paired primary denominator differs')
        pairs.append({'seed':seed,'train24_final_dev_cer':a['cer_percent'],'train72_final_dev_cer':b['cer_percent'],
                      'delta_cer_percentage_points':b['cer_percent']-a['cer_percent'],
                      'train24_final_dev_wer':a['wer_percent'],'train72_final_dev_wer':b['wer_percent'],
                      'train24_final_dev_exact':a['exact_sentences'],'train72_final_dev_exact':b['exact_sentences']})
    return {'models':models,'paired_final_dev16':pairs,
            'predeclared_followup_expansion_gate':gate([row['delta_cer_percentage_points'] for row in pairs],failures),
            'inference_rows':824,'failed_inference_rows':failures,'final_checkpoint_selected':3600,
            'used_best_intermediate_checkpoint':False,'new_main_optimizer_updates':14400,
            'same_updates_not_same_frames_or_wall_time':True,'service_weights_updated':False,
            'unseen_speaker_or_untouched_test_claim':False,
            'comparison_scope':'Fresh paired B_feature models; equal optimizer updates,24x300 versus72x100 clip exposures.'}


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def baseline(data,design):
    rows={row['id']:row for row in data['selection']['records']}
    anchors={}
    for sid in design['cohorts']['train24']: anchors.setdefault(rows[sid]['speaker_id'],sid)
    records=[]
    for sid in design['cohorts']['dev16']:
        row=rows[sid];anchor=anchors[row['speaker_id']]
        ref=unicodedata.normalize('NFC',row['text']).strip();pred=unicodedata.normalize('NFC',rows[anchor]['text']).strip()
        records.append({'id':sid,'anchor_id':anchor,'reference':ref,'prediction':pred,'status':'ok',
                        'metrics':{'character_edits':edit_distance(ref,pred),'reference_characters':len(ref),
                                   'word_edits':edit_distance(ref.split(),pred.split()),'reference_words':len(ref.split()),'exact':ref==pred}})
    return {'kind':'known_speaker_fixed_control_training_sentence','records':records,'aggregate':aggregate(records),
            'uses_dev_targets_to_choose_anchor':False,'is_deployable_lipreading_decoder':False,'neural_inference_rows':0}


def main():
    import protocol
    policy=protocol.verify_frozen()
    design=read(policy['step15_training_design']['path']);data=read(policy['step15_data_plan']['path'])
    paths=[BASE/'results'/f'evaluation_seed{seed}_{cohort}.json' for seed in SEEDS for cohort in COHORTS]
    refs=[bind(path) for path in paths];values=[read(path) for path in paths]
    for value in values: require(value['experiment_plan']==bind(protocol.POLICY),'Another experiment evaluated')
    result={'schema':'lipnet-step16-diversity-comparison-v1','status':'completed',
            'created_utc':datetime.now(timezone.utc).isoformat(),'experiment_plan':bind(protocol.POLICY),
            'evaluation_runs':refs,'generator':bind(__file__),**make_comparison(values,design),
            'known_speaker_baseline':baseline(data,design),
            'limitations':['The selected same-speaker dev16 is not an untouched test or webcam sample.',
                           'Only B_feature is studied; no new A/B architecture conclusion is justified.',
                           'Both datasets use the same LR schedule; this is not a separate LR ablation.',
                           'The practical five-percentage-point gate is not a significance test or deployment approval.']}
    require([bind(path) for path in paths]==refs,'Evaluation evidence changed during aggregation')
    protocol.verify_frozen()
    print(json.dumps(protocol.save_new(BASE/'results/comparison.json',result)))


if __name__=='__main__': main()
