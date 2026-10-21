"""Finalize the four executed runs only after independent state and output checks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import protocol

BASE=Path(__file__).resolve().parent
read,bind,require=protocol.read,protocol.bind,protocol.require


def main():
    policy=protocol.verify_frozen()
    comparison=read(BASE/'results/comparison.json')
    audit=read(BASE/'results/independent_audit.json')
    require(audit['status']=='passed' and comparison['status']=='completed' and
            audit['comparison']==bind(BASE/'results/comparison.json'),'Missing or stale independent audit')
    require(audit['main_optimizer_updates']==comparison['new_main_optimizer_updates']==14400 and
            audit['checkpoints_verified']==24 and audit['inference_rows']==comparison['inference_rows']==824,
            'Incomplete training/checkpoint/evaluation audit')
    require(audit['failed_inference_rows']==comparison['failed_inference_rows']==0,'Unresolved inference failures')
    require(audit['smoke_optimizer_updates']==12,'Changed smoke budget')
    require(comparison['final_checkpoint_selected']==3600 and not comparison['used_best_intermediate_checkpoint']
            and not comparison['service_weights_updated'],'Changed checkpoint selection or service scope')
    suites=ET.parse(BASE/'results/test_results.xml').getroot().findall('testsuite')
    require(suites and all(all(int(s.attrib.get(k,'0'))==0 for k in ('errors','failures','skipped')) for s in suites),'Tests not all passed')
    paths=['experiment_plan.json','plans/freeze_manifest.json','results/test_results.xml',
           'results/smoke_preflight.json','results/independent_preflight.json','results/comparison.json','results/independent_audit.json']
    training=[]
    for seed in (42,43):
        for cohort in ('train24','train72'):
            path=protocol.run_path(seed,cohort)/'run_result.json'
            run=read(path)
            require(run['completed_updates']==3600 and len(run['updates'])==3600,'Incomplete complete history')
            require(run['actual_optimizer_step_count_fully_known'] and
                    run['failed_or_unrecorded_update_attempts_total']==0 and
                    run['discarded_recorded_updates_from_abrupt_interruption']==0,
                    'Actual optimizer calls differ from the fixed budget or cannot be established')
            history=run['updates']
            windows={}
            for start,end in ((2101,2400),(2701,3000),(3301,3600)):
                selected=history[start-1:end]
                windows[f'{start}-{end}']={key:sum(row[key] for row in selected)/len(selected)
                                           for key in ('ctc_loss','feature_loss','gradient_norm_before_clipping')}
            training.append({'seed':seed,'cohort':cohort,'result':bind(path),'updates':3600,
                             'optimizer_seconds':sum(row['seconds_optimizer_update'] for row in history),
                             'student_frames_presented':sum(sum(row['frames']) for row in history),
                             'loss_windows':windows})
    result={'schema':'lipnet-step16-summary-v1','status':'completed_diversity_training_comparison',
            'created_utc':datetime.now(timezone.utc).isoformat(),'generator':bind(__file__),
            'artifacts':{name:bind(BASE/name) for name in paths},'training_runs':training,
            'tests_passed':sum(int(s.attrib['tests']) for s in suites),
            'main_optimizer_updates':14400,'smoke_optimizer_updates':12,'total_new_optimizer_updates':14412,
            'actual_optimizer_step_count_fully_known':True,'discarded_recorded_optimizer_updates':0,
            'production_checkpoints':24,'inference_rows':824,'failed_inference_rows':0,
            'paired_final_dev16':comparison['paired_final_dev16'],
            'predeclared_followup_expansion_gate':comparison['predeclared_followup_expansion_gate'],
            'independent_ctc_max_abs_difference':audit['max_ctc_abs_difference'],
            'downloads':0,'teacher_model_forwards':0,'raw_media_read':False,
            'service_weights_updated':False,'new_human_approvals_added':0,
            'same_updates_not_same_frames_or_wall_time':True,
            'unseen_speaker_or_webcam_readiness_established':False,
            'old_frozen_artifacts_preserved':True}
    protocol.verify_frozen()
    print(json.dumps(protocol.save_new(BASE/'results/step16_summary.json',result)))


if __name__=='__main__': main()
