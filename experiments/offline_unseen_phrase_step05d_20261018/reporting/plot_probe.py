"""Plot six fixed development-probe scores and the prior final training scores."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
DEV=BASE.parent/'results/aggregate_metrics.json'
TRAIN=ROOT/'experiments/offline_phrase_only_step05c_20261018/results/aggregate_metrics.json'
SEEDS=(42,43)
CONDITIONS=('normal','static','shuffled')
COLORS={42:'#1474b8',43:'#e87919'}


def ref(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}


def require(condition,message):
    if not condition:raise ValueError(message)


def extract():
    dev=json.loads(DEV.read_text(encoding='utf-8'))
    train=json.loads(TRAIN.read_text(encoding='utf-8'))
    require(dev['schema']=='offline-stage5d-fixed-probe-aggregate-v1' and dev['status']=='completed','Invalid dev aggregate')
    require(dev['checkpoint_count']==2 and dev['evaluation_count']==18 and dev['forward_calls']==18
            and dev['frames']==576 and dev['failures']==0 and dev['optimizer_updates']==0,'Invalid probe count')
    require(dev['exploratory_reused_development_probe'] is True and dev['generalization_established'] is False
            and dev['new_speaker_or_webcam_evaluation'] is False,'Exploratory scope changed')
    require(train['schema']=='offline-stage5c-evaluation-aggregate-v1' and train['status']=='completed','Invalid prior train aggregate')
    devkeys=[(r['seed'],r['completed_update'],r['condition']) for r in dev['diagnostics']]
    require(len(devkeys)==len(set(devkeys))==6 and set(devkeys)=={(s,1680,c) for s in SEEDS for c in CONDITIONS},'Missing or duplicate dev values')
    prior=[r for r in train['diagnostics'] if r['completed_update']==1680 and r['condition']=='normal']
    require(len(prior)==2 and sorted(r['seed'] for r in prior)==list(SEEDS),'Missing prior final train scores')
    points=[]
    for split,rows,denominator,count in [('dev',dev['diagnostics'],31,3),('train',prior,132,12)]:
        for row in rows:
            m=row['metrics'];edits=m['character_edits'];exact=m['exact_count']
            require(type(edits) is int and edits>=0 and m['reference_characters']==denominator and m['count']==count,
                    'Corpus counts differ')
            require(m['failure_count']==0 and type(exact) is int and 0<=exact<=count,'Incomplete or invalid exact count')
            require(math.isfinite(m['CER']) and math.isclose(m['CER'],edits/denominator,rel_tol=1e-12,abs_tol=1e-12),'CER arithmetic differs')
            points.append({'split':split,'seed':row['seed'],'completed_update':1680,'condition':row['condition'],
                'count':count,'character_edits':edits,'reference_characters':denominator,
                'CER_percent':100*edits/denominator,'exact_count':exact,'failure_count':0})
    require(len(points)==8,'Eight unique aggregate observations required')
    return points


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    outputs=[BASE/'unseen_phrase_probe.png',BASE/'plotted_points.json',BASE/'plot_verification.json']
    require(not any(p.exists() for p in outputs),'Existing outputs are immutable')
    points=extract()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'figure.facecolor':'white','savefig.facecolor':'white'}):
        fig,axes=plt.subplots(1,2,figsize=(12.5,6.1),sharey=True)
        width=.34
        for index,seed in enumerate(SEEDS):
            offset=-width/2 if index==0 else width/2
            dev=[next(p for p in points if p['split']=='dev' and p['seed']==seed and p['condition']==c) for c in CONDITIONS]
            bars=axes[0].bar(np.arange(3)+offset,[p['CER_percent'] for p in dev],width=width,color=COLORS[seed],zorder=3)
            axes[0].bar_label(bars,labels=[f"{p['CER_percent']:.2f}%" for p in dev],padding=4,fontsize=9)
            compare=[next(p for p in points if p['split']==split and p['seed']==seed and p['condition']=='normal') for split in ('train','dev')]
            bars=axes[1].bar(np.arange(2)+offset,[p['CER_percent'] for p in compare],width=width,color=COLORS[seed],zorder=3)
            axes[1].bar_label(bars,labels=[f"{p['CER_percent']:.2f}%\n{p['exact_count']}/{p['count']} exact" for p in compare],padding=4,fontsize=9)
        axes[0].set_title('A. Reused development phrases: temporal controls',pad=15)
        axes[1].set_title('B. Normal input: training and development cohorts',pad=15)
        axes[0].set_xticks(range(3),['Normal','Repeated\nmiddle frame','Shuffled\nframes'])
        axes[1].set_xticks(range(2),['Training: 12 phrases\n(prior stage5c result)','Development: 3 phrases\n(current stage5d probe)'])
        ymax=max(110,math.ceil((max(p['CER_percent'] for p in points)+15)/10)*10)
        for axis in axes:
            axis.set_ylim(0,ymax);axis.set_yticks(range(0,int(ymax)+1,20))
            axis.grid(axis='y',color='#dddddd',linewidth=.7)
            axis.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Corpus character error rate (CER, %) - lower is better')
        fig.suptitle('Frozen phrase-trained models: three-phrase development probe',fontsize=15,y=.975)
        fig.legend([Patch(color=COLORS[s],label=f'Seed {s}') for s in SEEDS],
                   [f'Seed {s}' for s in SEEDS],loc='upper center',bbox_to_anchor=(.5,.923),ncol=2,frameon=False)
        caption=('Both models are fixed at update 1,680. All development conditions have 0/3 exact phrases.\n'
                 'The same three development phrases were evaluated earlier in the project; this is an exploratory reused-development probe.\n'
                 'Training and development use different utterances and corpus sizes. These scores do not establish unseen-speaker or webcam accuracy.')
        fig.text(.5,.035,caption,ha='center',va='bottom',fontsize=9,color='#444444',linespacing=1.55)
        fig.subplots_adjust(left=.09,right=.98,top=.78,bottom=.28,wspace=.15)
        fig.savefig(outputs[0],dpi=180);plt.close(fig)
    data={'schema':'offline-stage5d-plotted-points-v1','sources':{'dev':ref(DEV),'train':ref(TRAIN)},'points':points,
        'unique_aggregate_observations':8,'plotted_scalar_CER_values':10,
        'dev_normal_repeated_across_panels':True,'panel_A_dev_conditions':6,'panel_B_train_dev_normal':4,
        'fixed_checkpoint':1680,'checkpoint_selection':False,'CER_unit':'percent derived from summed edits / summed reference characters',
        'different_corpus_sizes':{'train':12,'dev':3},'new_neural_calls':0,'source_mutations':0}
    outputs[1].write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    checks={'schema':'offline-stage5d-plot-verification-v1','status':'passed_numeric_and_scope_checks',
        'created_utc':datetime.now(timezone.utc).isoformat(),'script':ref(Path(__file__)),'sources':data['sources'],
        'points':ref(outputs[1]),'png':ref(outputs[0]),'all_integer_error_denominators_recomputed':True,
        'all_fixed_seed_condition_pairs_present':True,'train_dev_cohorts_explicit':True,'CER_not_capped_at_100':True,
        'visual_inspection':'pending_separate_agent_view','new_neural_calls':0,'source_mutations':0}
    outputs[2].write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':checks['status'],'png':str(outputs[0]),'unique_observations':8}))


if __name__=='__main__':main()
