from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('step16_summary_test_subject',Path(__file__).with_name('summarize.py'))
subject=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(subject)


def fixture():
    design={'cohorts':{'train24':[f't{i}' for i in range(24)],'train72':[f't{i}' for i in range(72)],
                       'dev16':[f'd{i}' for i in range(16)]},
            'evaluation':{'historical_probe14_ids':[f'p{i}' for i in range(14)]}}
    result=[]
    for seed in (42,43):
        for cohort in ('train24','train72'):
            edits=8 if cohort=='train24' else 6
            value={'schema':'lipnet-step16-diversity-evaluation-v1','seed':seed,'cohort':cohort,'milestones':[]}
            for update in (1200,2400,3000,3600):
                conditions={}
                for cond in (('normal','static','shuffled') if update==3600 else ('normal',)):
                    ids=((design['cohorts'][cohort] if update in (2400,3600) else [])+design['cohorts']['dev16']+
                         (design['evaluation']['historical_probe14_ids'] if update==3600 else [])) if cond=='normal' else design['cohorts']['dev16']
                    rows=[{'id':sid,'status':'ok','prediction':'예측','metrics':{'character_edits':edits,'reference_characters':10,
                           'word_edits':1,'reference_words':2,'exact':False}} for sid in ids]
                    groups={'dev16':{'aggregate':subject.aggregate([r for r in rows if r['id'].startswith('d')])}}
                    if cond=='normal' and update in (2400,3600):
                        groups.update(train_native={'aggregate':subject.aggregate([r for r in rows if r['id'].startswith('t')])},
                                      train_common24={'aggregate':subject.aggregate([r for r in rows if r['id'] in design['cohorts']['train24']])})
                    conditions[cond]={'records':rows,'groups':groups,'per_speaker':{}}
                value['milestones'].append({'completed_updates':update,'conditions':conditions,
                                             'model_state_unchanged':True,'model_state_before':'same','model_state_after':'same'})
            value['record_count']=2*len(design['cohorts'][cohort])+110
            result.append(value)
    return result,design


@pytest.mark.parametrize('deltas,failures,expected',[([-5,-5],0,True),([-1,-9],0,True),
    ([0,-12],0,False),([1,-15],0,False),([-4,-4],0,False),([-5,-5],1,False)])
def test_predeclared_gate_requires_both_seeds_and_no_failures(deltas,failures,expected):
    assert subject.gate(deltas,failures)['passed'] is expected


def test_four_runs_824_rows_and_fixed_final():
    values,design=fixture();result=subject.make_comparison(values,design)
    assert result['inference_rows']==824 and result['failed_inference_rows']==0
    assert [p['delta_cer_percentage_points'] for p in result['paired_final_dev16']]==[-20,-20]
    assert result['predeclared_followup_expansion_gate']['passed'] is True
    assert result['final_checkpoint_selected']==3600 and result['used_best_intermediate_checkpoint'] is False


@pytest.mark.parametrize('change',['drop_run','duplicate_run','drop_row','reorder_row','missing_condition','missing_milestone','changed_state','changed_metrics'])
def test_scope_or_metric_corruption_rejected(change):
    values,design=fixture()
    if change=='drop_run': values.pop()
    elif change=='duplicate_run': values[-1]=deepcopy(values[0])
    elif change=='drop_row': values[0]['milestones'][0]['conditions']['normal']['records'].pop()
    elif change=='reorder_row': values[0]['milestones'][0]['conditions']['normal']['records'].reverse()
    elif change=='missing_condition': values[0]['milestones'][-1]['conditions'].pop('static')
    elif change=='missing_milestone': values[0]['milestones'].pop(0)
    elif change=='changed_state': values[0]['milestones'][-1]['model_state_after']='changed'
    else: values[0]['milestones'][0]['conditions']['normal']['groups']['dev16']['aggregate']['cer_percent']=0
    with pytest.raises(ValueError): subject.make_comparison(values,design)


def test_failed_row_preserves_denominator_and_blocks_gate():
    values,design=fixture()
    content=values[0]['milestones'][0]['conditions']['normal']
    row=content['records'][0];row.update(status='failed',prediction='')
    row['metrics'].update(character_edits=10,word_edits=2)
    content['groups']['dev16']['aggregate']=subject.aggregate(content['records'])
    result=subject.make_comparison(values,design)
    assert result['inference_rows']==824 and result['failed_inference_rows']==1
    assert result['predeclared_followup_expansion_gate']['passed'] is False


def test_baseline_uses_first_training_sentence_without_target_selection():
    rows=[{'id':'train_first','speaker_id':'s','text':'가나'}, {'id':'train_best','speaker_id':'s','text':'정답'},
          {'id':'dev','speaker_id':'s','text':'정답'}]
    design={'cohorts':{'train24':['train_first','train_best'],'dev16':['dev']}}
    result=subject.baseline({'selection':{'records':rows}},design)
    assert result['records'][0]['prediction']=='가나' and result['aggregate']['exact_sentences']==0
    assert result['uses_dev_targets_to_choose_anchor'] is False
