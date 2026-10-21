"""Reproduce the step03 independent input/result audit without any model forward.

Archived from the two inline Python commands actually executed for the original
2026-09-24 audit, combined here for reproducibility. The original audit JSON is
unchanged. This archival script has not been re-executed as part of archiving.
Reads prepared crops/traces, saved feature NPZ and text/provenance files only;
does not reopen original G: videos, checkpoints or test50 payloads.
Run without Python -O so assertions remain enabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit_inputs(base: Path, manifest_path: Path) -> dict:
    """Reproduce the separately executed input audit without cloud-source reads."""
    import cv2
    import numpy as np
    from collections import Counter
    import math
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    plan = json.loads((base / "step03_eval_plan.json").read_text(encoding="utf8"))
    sources = json.loads((base / "step03_source_plan.json").read_text(encoding="utf8"))
    qa = json.loads((manifest_path.parent / "input_qa.json").read_text(encoding="utf8"))
    assert sha(manifest_path) == "a03a55eb67b78e92d2dab76a491bab2b4f2acfbf81e05c218ce396c8e7018613" == qa["manifest_sha256"]
    assert sha(base / "step03_source_plan.json") == plan["source_plan_sha256"] == manifest["source_plan_sha256"]
    assert sha(base / "step03_eval_plan.json") == manifest["evaluation_plan"]["sha256"]
    assert sha(base / "reference_profile_96_v1.json") == plan["reference_profile_sha256"] == manifest["reference_profile_sha256"]
    assert sha(base / "prepare_step03_inputs.py") == manifest["preparation_script_sha256"]
    assert sha(base / "automatic_roi.py") == manifest["automatic_roi_script_sha256"]
    assert sha(base / "corrected_search.py") == plan["corrected_search_sha256"]
    references = {record["id"]: record for record in sources["records"]}
    lookup = {(record["id"], record["arm"]): record for record in manifest["records"]}
    assert len(manifest["records"]) == len(lookup) == 40
    assert set(lookup) == {(sid, arm) for sid in plan["source_ids"] for arm in plan["arms"]}
    assert all(record["split"] == "val" for record in manifest["records"])
    assert not manifest["audio_used"] and not manifest["test_payloads_read"]
    frames, statuses = Counter(), Counter()
    offsets, timestamps = [], []
    previous = 0
    for sid in plan["source_ids"]:
        annotation, automatic = (lookup[sid, arm] for arm in plan["arms"])
        assert annotation["source_selected_frames_sha256"] == automatic["source_selected_frames_sha256"]
        assert annotation["trace"] == automatic["trace"]
        assert annotation["start_frame"] == automatic["start_frame"]
        assert annotation["stop_frame_exclusive"] == automatic["stop_frame_exclusive"]
        assert annotation["text"] == automatic["text"] == references[sid]["text"]
        assert annotation["status"] == automatic["status"] == "ok"
        fps = annotation["native_fps"]
        start = math.ceil(references[sid]["start_time"] * fps - 1e-9)
        stop = math.ceil(references[sid]["end_time"] * fps - 1e-9)
        assert (start, stop) == (annotation["start_frame"], annotation["stop_frame_exclusive"])
        assert sha(annotation["trace"]["path"]) == annotation["trace"]["sha256"]
        trace = json.loads(Path(annotation["trace"]["path"]).read_text(encoding="utf8"))
        assert len(trace["frames"]) == stop - start
        for index, frame in enumerate(trace["frames"], start):
            assert frame["source_frame_index"] == index
            assert frame["decoder_position_after_read"] == index + 1
            assert frame["decoder_timestamp_available"]
            assert abs(frame["source_timestamp_seconds"] - index / fps) < 1e-10
            assert abs(frame["decoder_timestamp_seconds"] - index / fps) < 1e-10
            assert not frame["automatic_failed"]
            statuses[frame["automatic_frame_record"]["status"]] += 1
            timestamps.append(abs(frame["decoder_timestamp_seconds"] - index / fps))
        offsets += [start / fps - annotation["start_time"], stop / fps - annotation["end_time"]]
        for record in (annotation, automatic):
            assert sha(record["path"]) == record["sha256"]
            capture, parts = cv2.VideoCapture(record["path"]), []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    parts.append(frame)
            finally:
                capture.release()
            pixels = np.stack(parts)
            assert list(pixels.shape) == record["shape"] == [stop - start, 96, 96, 3]
            assert len(pixels) == record["frames"] == record["expected_frames"]
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == record["decoded_bgr_stream_sha256"]
            assert pixels.dtype == np.uint8
            assert abs(float(pixels.mean()) - record["mean_before_normalization"]) < 1e-8
            assert abs(float(pixels.std(ddof=1)) - record["std_before_normalization_ddof1"]) < 1e-8
            frames[record["arm"]] += len(pixels)
            old = record.get("previous_partial_run_comparison", {})
            if old.get("comparison_performed"):
                assert sha(old["path"]) == sha(record["path"]) == old["sha256"]
                previous += 1
    assert frames == Counter({"annotation_bbox": 2572, "automatic_mediapipe": 2572})
    assert statuses == Counter(detected=2572)
    return {
        "status": "passed", "manifest_sha256": sha(manifest_path),
        "records": 40, "utterances": 20, "source_files": 4,
        "decoded_frames_by_arm": dict(frames), "automatic_statuses": dict(statuses),
        "all_encoded_and_decoded_hashes_verified": True,
        "all_native_bounds_recomputed": True,
        "all_trace_sha_and_frame_positions_verified": True,
        "source_pixels_not_reread": True,
        "max_abs_decoder_minus_cfr_seconds": max(timestamps),
        "boundary_offset_seconds": [min(offsets), max(offsets)],
        "prior_completed_files_byte_reproduced": previous,
        "reference_characters": sum(len(row["text"]) for row in references.values()),
        "reference_words": sum(len(row["text"].split()) for row in references.values()),
        "decoder_library": cv2.__version__,
    }


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=base / "results/step03.json")
    parser.add_argument("--manifest", type=Path, default=base / "inputs_step03_recovery02/manifest.json")
    parser.add_argument("--output", type=Path, default=base / "results/step03_metrics_audit_rerun.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior audits: choose a new --output path")
    import csv,hashlib,json,math,statistics,sys,unicodedata
    from collections import Counter
    from itertools import groupby
    from datetime import datetime,timezone
    import numpy as np
    b=Path(__file__).resolve().parent
    sys.path.insert(0,str(b))
    from summarize_legacy_step03 import independent_counts
    sys.path.insert(0,str(b/'upstream'))
    from avsr.utils.korean_funcs import grp2char
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result_path=args.result.resolve()
    result=json.loads(result_path.read_text(encoding='utf8'))
    if result['status']!='completed':
     print(json.dumps({'status':result['status'],'records':len(result['records'])}));sys.exit(7)
    manifest_path=args.manifest.resolve()
    INPUT_AUDIT=audit_inputs(b,manifest_path)
    m=json.loads(manifest_path.read_text(encoding='utf8'));p=json.loads((b/'step03_eval_plan.json').read_text(encoding='utf8'));source=json.loads((b/'step03_source_plan.json').read_text(encoding='utf8'))
    assert result['plan']==p
    assert sha(manifest_path)==result['input_manifest_sha256']==INPUT_AUDIT['manifest_sha256']
    assert sha(b/'step03_eval_plan.json')==result['plan_sha256']
    assert sha(b/'step03_source_plan.json')==result['source_plan_sha256']==p['source_plan_sha256']
    assert sha(b/'evaluate_step03.py')==result['runner_sha256']
    assert sha(b/'evaluation_metrics.py')==result['metrics_code_sha256']
    assert sha(b/'diagnose_step02.py')==result['read_tensor_code_sha256']
    assert sha(b/'corrected_search.py')==p['corrected_search_sha256']
    assert not result['training_run'] and not result['audio_used'] and not result['test50_used']
    refs={r['id']:r for r in source['records']};inputs={(r['id'],r['arm']):r for r in m['records']}
    records=result['records'];assert len(records)==40
    assert len({(r['id'],r['arm']) for r in records})==40
    assert {(r['id'],r['arm']) for r in records}==set(inputs)
    assert [(r['id'],r['arm']) for r in records]==[(sid,a) for a in p['arms'] for sid in p['source_ids']]
    with (b/'upstream/vocabulary/kor_grapheme.csv').open(encoding='utf-8-sig',newline='') as stream:
     vocab={int(r['id']):r['char'] for r in csv.DictReader(stream)}
    assert len(vocab)==72
    feature_count=metric_count=string_count=ctc_count=0
    search_depth=[];search_caps=[];output_status=Counter();paired_results=[];detailed=[]
    for r in records:
     inp=inputs[r['id'],r['arm']];ref=refs[r['id']]
     assert r['reference']==inp['text']==ref['text']
     assert r['frames']==inp['frames']
     assert r['speaker_id']==ref['speaker_id']
     if r['input_status']=='ok':
      assert sha(inp['path'])==inp['sha256']==r['input_sha256']
     if 'feature_file' in r:
      assert sha(r['feature_file'])==r['feature_file_sha256']
      with np.load(r['feature_file'],allow_pickle=False) as stored:
       features=stored['features'];logits=stored['ctc_logits'] if 'ctc_logits' in stored else None
       assert list(features.shape)==r['feature_shape'][1:]==[inp['frames'],256]
       assert features.dtype==np.float32 and np.isfinite(features).all()
       if logits is not None:
        assert logits.shape==(inp['frames'],72) and logits.dtype==np.float32 and np.isfinite(logits).all()
        path=logits.argmax(-1).tolist()
        assert path==r['ctc_frame_path']
        assert r['ctc_stats']['token_counts']=={str(k):v for k,v in Counter(path).items()}
        assert r['ctc_stats']['blank_fraction']==path.count(3)/len(path)
        assert r['ctc_stats']['first_eos_frame']==(path.index(2) if 2 in path else None)
        ctc=[token for token,group in groupby(path) if token!=3]
        if r['outputs']['ctc_greedy']['status']=='ok':
         assert ctc==r['outputs']['ctc_greedy']['token_ids'];ctc_count+=1
      feature_count+=1
     for method in p['methods']:
      o=r['outputs'][method];metric_count+=1;output_status[o['status']]+=1
      audit_counts=independent_counts(r['reference'],o['prediction'])
      assert audit_counts==o['metrics'],(r['id'],r['arm'],method,audit_counts,o['metrics'])
      if o['status']!='ok':
       assert o['prediction']==''
      else:
       tokens=o['token_ids'];assert all(type(t)==int and 0<=t<72 for t in tokens)
       prefix=tokens[:tokens.index(2)] if 2 in tokens else tokens
       graphemes=''.join(vocab[t] for t in prefix if t!=3)
       assert graphemes==o['graphemes'];assert grp2char(graphemes).strip()==o['prediction']
       assert o['eos_index']==(tokens.index(2) if 2 in tokens else None)
       string_count+=1
       if method=='corrected_hybrid':
        assert tokens[-1]==2 and tokens.count(2)==1 and not set(tokens)&{0,1}
        assert len(tokens)<=o['max_search_length']<=199
        assert o['search_cap_reached']==(o['max_search_length']>=199)
        search_depth.append(o['max_search_length'])
        if o['search_cap_reached']:search_caps.append({'id':r['id'],'arm':r['arm']})
      detailed.append({'id':r['id'],'arm':r['arm'],'method':method,'status':o['status'],'counts':audit_counts})
    
    def independent_aggregate(rows):
     counts=[independent_counts(r['reference'],o['prediction']) for r,o in rows]
     keys=['character_edits','reference_characters','word_edits','reference_words','normalized_character_edits','normalized_reference_characters']
     z={k:sum(x[k] for x in counts) for k in keys}
     z.update(clips=len(rows),failed_inputs=sum(o['status']!='ok' for r,o in rows),exact_sentences=sum(x['exact'] for x in counts),normalized_exact_sentences=sum(x['normalized_exact'] for x in counts))
     z.update(cer_percent=100*z['character_edits']/z['reference_characters'],wer_percent=100*z['word_edits']/z['reference_words'],normalized_cer_percent=100*z['normalized_character_edits']/z['normalized_reference_characters'])
     return z
    
    aggregates={}
    for arm in p['arms']:
     selected=[r for r in records if r['arm']==arm]
     for method in p['methods']:
      key=f'{arm}/{method}';rows=[(r,r['outputs'][method]) for r in selected];z=independent_aggregate(rows)
      for k,v in z.items():assert result['aggregate'][key][k]==v,(key,k)
      assert z['clips']==20 and z['reference_characters']==667 and z['reference_words']==189
      z['per_speaker']={}
      for speaker in sorted({r['speaker_id'] for r in selected}):
       a=independent_aggregate([(r,o) for r,o in rows if r['speaker_id']==speaker]);assert a==result['aggregate'][key]['per_speaker'][speaker];z['per_speaker'][speaker]=a
      times=[r['feature_timing']['seconds']+o['decoder_timing']['seconds'] for r,o in rows if o['status']=='ok']
      t={'n':len(times),'min_seconds':min(times),'median_seconds':statistics.median(times),'max_seconds':max(times),'mean_seconds':statistics.mean(times)} if times else None
      assert t==result['aggregate'][key]['model_timing'];z['model_timing']=t
      aggregates[key]=z
    for sid in p['source_ids']:
     pair={r['arm']:r for r in records if r['id']==sid}
     a=pair['annotation_bbox']['outputs']['corrected_hybrid']['metrics']['character_edits'];d=pair['automatic_mediapipe']['outputs']['corrected_hybrid']['metrics']['character_edits']
     paired_results.append({'id':sid,'annotation_edits':a,'automatic_edits':d,'delta_auto_minus_annotation':d-a})
    audit={'schema':'olkavs-step03-independent-metrics-audit-v1','status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
     'result_path':str(result_path),'result_sha256':sha(result_path),'input_audit':INPUT_AUDIT,
     'verified':{'plan_and_code_hash_chain':True,'frozen20_ids_both_arms':True,'full80_output_metric_records':metric_count,'independent_full_matrix_levenshtein':True,'per_speaker_aggregates':True,'pooled_aggregates':True,'timing_statistics':True,'feature_files_hash_and_finite_shape':feature_count,'ctc_logits_argmax_collapse_reproduced':ctc_count,'token_dictionary_and_string_reconstruction':string_count,'model_not_rerun':True,'source_videos_not_reread':True,'training_or_test50_payload_not_read':True},
     'audit_dependency_hashes':{'summarize_legacy_step03.py':sha(b/'summarize_legacy_step03.py'),'upstream/avsr/utils/korean_funcs.py':sha(b/'upstream/avsr/utils/korean_funcs.py'),'upstream/vocabulary/kor_grapheme.csv':sha(b/'upstream/vocabulary/kor_grapheme.csv')},
     'output_status_counts':dict(output_status),'search':{'minimum_max_search_length':min(search_depth),'maximum_max_search_length':max(search_depth),'cap_reached':search_caps},
     'aggregate':aggregates,'paired_hybrid_character_edits':paired_results,'per_output_independent_counts':detailed,
     'limits':['Development20 was repeatedly examined in the former project; not untouched final test.','Official pretraining overlap unknown.','Source frame hashes matched across paired arms but original cloud pixels were not independently reread.','Hangul composition reused the frozen official pure text helper; edit distances and CTC collapse were independently implemented.','No model forward was rerun; stored finite features and CTC logits were checked.','File-based labelled utterance evaluation does not measure webcam segment detection or total live latency.']}
    out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':audit['status'],'output':str(out),'records':metric_count,'aggregate':aggregates,'search':audit['search']},ensure_ascii=False,indent=2))
    


if __name__ == "__main__":
    main()
