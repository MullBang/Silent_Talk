"""Plot the fixed phrase-only training evaluations, without model or pixel reads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
STEPS = (420, 840, 1260, 1680)
SEEDS = (42,43)
CONDITIONS = ('normal','static','shuffled')
COLORS = {42:'#1474b8',43:'#e87919'}


def require(condition, message):
    if not condition: raise ValueError(message)


def ref(path):
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}


def extract_points(aggregate):
    require(aggregate['schema']=='offline-stage5c-evaluation-aggregate-v1' and aggregate['status']=='completed','Wrong aggregate')
    require(aggregate['checkpoint_count']==8 and aggregate['evaluation_count']==144 and aggregate['forward_calls']==144,
            'Fixed evaluation counts changed')
    require(aggregate['frames']==5772 and aggregate['failures']==0,'Incomplete evaluation')
    require(aggregate['dev_evaluated'] is False and aggregate['generalization_measured'] is False
            and aggregate['automatic_sentence_expansion'] is False,'Training diagnostic scope changed')
    expected={(seed,step,'normal') for seed in SEEDS for step in STEPS}|{
        (seed,1680,condition) for seed in SEEDS for condition in ('static','shuffled')}
    records=aggregate['diagnostics']
    keys=[(r['seed'],r['completed_update'],r['condition']) for r in records]
    require(len(keys)==len(set(keys))==12 and set(keys)==expected,'Missing or duplicate fixed observations')
    points=[]
    for row in sorted(records,key=lambda r:(r['seed'],r['completed_update'],CONDITIONS.index(r['condition']))):
        metrics=row['metrics']
        edits,chars,exact=metrics['character_edits'],metrics['reference_characters'],metrics['exact_count']
        require(type(edits) is int and edits>=0 and type(chars) is int and chars==132,'Invalid corpus denominator')
        require(metrics['count']==12 and metrics['failure_count']==0 and type(exact) is int and 0<=exact<=12,
                'Invalid training cohort')
        require(math.isfinite(metrics['CER']) and math.isclose(metrics['CER'],edits/chars,abs_tol=1e-12,rel_tol=1e-12),
                'CER does not match integer errors / reference characters')
        points.append({'seed':row['seed'],'completed_update':row['completed_update'],'condition':row['condition'],
            'count':12,'character_edits':edits,'reference_characters':chars,'CER':metrics['CER'],
            'CER_percent':100*edits/chars,'exact_count':exact,'failure_count':0})
    gates={s:all(p['exact_count']==12 and p['CER']==0 for p in points if p['seed']==s
                 and p['completed_update'] in (1260,1680) and p['condition']=='normal') for s in SEEDS}
    require({g['seed']:g['passed'] for g in aggregate['fit_gate']['per_seed']}==gates,'Per-seed gate differs')
    require(aggregate['fit_gate']['passed']==all(gates.values()),'Combined gate differs')
    return points,gates


def plot(source,output_dir):
    source,output_dir=source.resolve(),output_dir.resolve()
    require(output_dir.is_relative_to(BASE),'Output must remain under this reporting directory')
    points,gates=extract_points(json.loads(source.read_text(encoding='utf-8')))
    output_dir.mkdir(parents=True,exist_ok=True)
    png=output_dir/'phrase_only_learning_and_controls.png'
    point_file=output_dir/'plotted_points.json'
    verification_file=output_dir/'plot_verification.json'
    require(not any(p.exists() for p in (png,point_file,verification_file)),'Existing plot outputs are immutable')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                         'savefig.facecolor':'white','figure.facecolor':'white'}):
        fig,axes=plt.subplots(2,2,figsize=(13,8.8))
        normal=[p for p in points if p['condition']=='normal']
        for seed in SEEDS:
            series=[p for p in normal if p['seed']==seed]
            x=[p['completed_update'] for p in series]
            for axis,field in ((axes[0,0],'CER_percent'),(axes[0,1],'exact_count')):
                y=[p[field] for p in series]
                axis.plot(x,y,color=COLORS[seed],label=f'Seed {seed}',linestyle='-' if seed==42 else '--',
                          linewidth=2,marker='o' if seed==42 else 's',markersize=5,
                          markerfacecolor=COLORS[seed] if seed==42 else 'white',zorder=3)
        for axis in axes[0]:
            axis.axvspan(1260,1680,color='#eeeeee',alpha=.65,zorder=0)
            axis.set_xlim(360,1740); axis.set_xticks(STEPS); axis.set_xlabel('Optimizer updates')
            axis.grid(axis='y',color='#dddddd',linewidth=.7)
        axes[0,0].set_title('A. Training phrases: character error rate',pad=12)
        axes[0,1].set_title('B. Training phrases: exact matches',pad=12)
        axes[0,0].set_ylabel('Corpus CER (%) - lower is better')
        axes[0,0].set_ylim(0,max(100,math.ceil((max(p['CER_percent'] for p in normal)+5)/20)*20))
        axes[0,1].set_ylabel('Exact phrases out of 12')
        axes[0,1].set_yticks(range(0,13,2)); axes[0,1].set_ylim(-.3,13)
        axes[0,1].axhline(12,linestyle=':',color='#666666',linewidth=1,zorder=1)

        final=[p for p in points if p['completed_update']==1680]
        positions=np.arange(3); width=.34
        for seed_index,seed in enumerate(SEEDS):
            series=[next(p for p in final if p['seed']==seed and p['condition']==c) for c in CONDITIONS]
            x=positions+(-width/2 if seed_index==0 else width/2)
            for axis,field in ((axes[1,0],'CER_percent'),(axes[1,1],'exact_count')):
                values=[p[field] for p in series]
                bars=axis.bar(x,values,width=width,color=COLORS[seed],edgecolor='white',linewidth=.7,zorder=3)
                labels=[f'{v:.2f}%' if field=='CER_percent' else f'{v}/12' for v in values]
                axis.bar_label(bars,labels=labels,padding=4,fontsize=9)
        for axis in axes[1]:
            axis.set_xticks(positions,['Normal','Repeated middle frame','Shuffled frames'])
            axis.tick_params(axis='x',labelsize=9)
            axis.grid(axis='y',color='#dddddd',linewidth=.7)
        axes[1,0].set_title('C. Final update 1,680: temporal controls / CER',pad=12)
        axes[1,1].set_title('D. Final update 1,680: temporal controls / exact',pad=12)
        axes[1,0].set_ylabel('Corpus CER (%) - lower is better')
        axes[1,0].set_ylim(0,max(100,math.ceil((max(p['CER_percent'] for p in final)+12)/20)*20))
        axes[1,1].set_ylabel('Exact phrases out of 12')
        axes[1,1].set_yticks(range(0,13,2)); axes[1,1].set_ylim(0,13.8)
        for axis in axes.flat: axis.spines[['top','right']].set_visible(False)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.suptitle('Phrase-only visual lipreading: fixed 12-phrase learning diagnostic',fontsize=15,y=.98)
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,frameon=False)
        gate_text='; '.join(f'Seed {seed}: '+('pass' if gates[seed] else 'not met') for seed in SEEDS)
        caption=('All results reuse the same 12 training phrases (11 unique texts, 8 speakers); no development/test evaluation.\n'
                 'Four measured checkpoints only; lines connect observations. Gray band: both checkpoints must reach 12/12 and CER 0.\n'
                 f'Fixed train-fit gate - {gate_text}. Training fit does not establish new-sentence or webcam accuracy.')
        fig.text(.5,.025,caption,ha='center',va='bottom',fontsize=9,color='#444444',linespacing=1.5)
        fig.subplots_adjust(left=.085,right=.975,top=.85,bottom=.18,hspace=.43,wspace=.25)
        fig.savefig(png,dpi=180); plt.close(fig)
    point_data={'schema':'offline-stage5c-plotted-points-v1','source':ref(source),'points':points,
        'unique_observations':12,'plotted_scalar_values':28,'best_checkpoint_selection':False,'smoothing':False,
        'panels':{'A':'8 normal CER values','B':'8 normal exact-count values',
                  'C':'6 final CER values (2 overlap with A)','D':'6 final exact-count values (2 overlap with B)'},
        'normal_final_repeated_across_panels':True,'gate_by_seed':gates}
    point_file.write_text(json.dumps(point_data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verification={'schema':'offline-stage5c-plot-verification-v1','status':'passed_numeric_and_scope_checks',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source':ref(source),'script':ref(Path(__file__)),
        'png':ref(png),'points':ref(point_file),'unique_observations':12,'plotted_scalar_values':28,
        'corpus_denominators_checked':True,'gate_recomputed':True,'CER_not_capped_at_100':True,
        'new_model_calls':0,'source_mutations':0,'visual_inspection':'pending_separate_agent_view'}
    verification_file.write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'png':str(png),'status':verification['status'],'points':12}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate',type=Path,default=BASE.parent/'results/aggregate_metrics.json')
    parser.add_argument('--output-dir',type=Path,default=BASE)
    args=parser.parse_args(); print(json.dumps(plot(args.aggregate,args.output_dir)))
