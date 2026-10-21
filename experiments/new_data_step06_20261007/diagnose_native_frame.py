"""One explicitly requested extra source read for C646 visual texture diagnosis.

Preserves the frozen preparation. No denoising, sharpening, model, audio read,
threshold change, transcript decision, or training approval. Uses one audited
video member, saves lossless native pixels and unresized crops, compares the
original preparation's raw-frame and96px ROI pixel hashes.
"""
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new120_frozen_preparation_for_diagnosis',BASE/'prepare_media.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PLAN_SHA='1e866dc03e70161eabc16b5fe5b0afc88eae10eefcafbc9e06718691091aee3d'
MANIFEST_SHA='fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8'
STEM='lip_J_6_M_03_C646_A_004'


def main():
    started=time.perf_counter()
    _,sources,builder,_,_=m.verify_plan(PLAN_SHA)
    manifest_path=BASE/'results/media_manifest.json'
    m.require(m.sha(manifest_path)==MANIFEST_SHA,'Frozen media manifest differs')
    manifest=m.read(manifest_path)
    source=next(s for s in sources['sources'] if s['stem']==STEM)
    clip=next(c for c in manifest['clips'] if c['source_stem']==STEM)
    sample=next(s for s in clip['roi_samples'] if s['position']=='middle')
    folder=BASE/'diagnosis'/STEM
    folder.mkdir(parents=True,exist_ok=False)
    plan={'schema':'lipnet-new120-native-frame-diagnosis-plan-v1','source_plan':m.bind(m.SOURCE_PLAN),
          'media_manifest':m.bind(manifest_path),'media_plan':m.bind(m.PLAN),'code':m.bind(Path(__file__)),
          'source_stem':STEM,'clip_id':clip['id'],'chosen_position':'middle',
          'selection_policy':'first originally selected sentence for C646_A004, middle student frame',
          'video_member':source['members']['video'],'sample':sample,
          'additional_archive_member_materializations':1,'audio_read':False,
          'question':'Is the visible skin texture present in the native decoded frame before ROI resizing?',
          'training_ready':False,'thresholds_changed':False,'original_writes':False}
    m.save(folder/'diagnosis_plan.json',plan)
    transport_source={'stem':source['stem'],'members':{'video':source['members']['video']}}
    with builder.transport.materialize_source(transport_source,folder/'temporary') as paths:
        capture=cv2.VideoCapture(str(paths['video']))
        try:
            m.require(capture.isOpened(),'Cannot open verified native MP4')
            index=sample['native_frame_index'];fps=float(capture.get(cv2.CAP_PROP_FPS))
            m.require(abs(fps-source['video_fps'])<=1e-9,'Native FPS differs')
            m.require(capture.set(cv2.CAP_PROP_POS_FRAMES,index),'Native seek failed')
            ok,frame=capture.read()
            m.require(ok and frame is not None,'Native frame decode failed')
            pts=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            m.require(abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-index-1)<=1e-6
                      and abs(pts-index/fps)<=.001,'Native frame index/PTS differs')
            raw_binding=m.array_binding(frame)
            m.require(raw_binding==sample['raw_frame_array'],'Re-decoded original frame pixels differ from frozen preparation')
            x1,y1,x2,y2=sample['crop_xyxy']
            crop=frame[y1:y2,x1:x2].copy()
            roi=cv2.resize(crop,(96,96),interpolation=cv2.INTER_LINEAR)
            m.require(m.array_binding(roi)==sample['roi_array'],'96px ROI pixels differ from frozen preparation')
            bx1,by1,bx2,by2=sample['bbox_xyxy']
            # Annotation pixels are integral in these labels; reject any ambiguity.
            m.require(all(float(v).is_integer() for v in [bx1,by1,bx2,by2]),'Noninteger exact annotation crop')
            annotation=frame[int(by1):int(by2),int(bx1):int(bx2)].copy()
            artifacts={'native_full_frame':m.write_png(folder/'native_full_frame.png',frame),
                       'unresized_margin_crop':m.write_png(folder/'unresized_margin_crop.png',crop),
                       'unresized_annotation_crop':m.write_png(folder/'unresized_annotation_crop.png',annotation),
                       'reproduced96px_roi':m.write_png(folder/'reproduced96px_roi.png',roi)}
        finally:
            capture.release()
    result={'schema':'lipnet-new120-native-frame-diagnosis-result-v1','plan':m.bind(folder/'diagnosis_plan.json'),
            'source_stem':STEM,'clip_id':clip['id'],'native_frame_index':index,'decoded_pts_seconds':pts,
            'video_member_sha256':source['members']['video']['sha256'],'bbox_xyxy':sample['bbox_xyxy'],
            'crop_xyxy':sample['crop_xyxy'],'raw_frame_array':raw_binding,'artifacts':artifacts,
            'native_frame_pixels_match_original_preparation':True,'roi96_pixels_match_original_preparation':True,
            'temporary_originals_cleaned':not any((folder/'temporary').iterdir()),
            'extra_materializations':1,'elapsed_seconds':time.perf_counter()-started,
            'limits':['Pixel equality rules out a different seek/crop in this repeat; it does not distinguish camera noise, compression artifacts and decoder behavior.',
                      'All saved images preserve the selected native pixels without denoising or sharpening; the96px image alone is resized.',
                      'This one-frame inspection does not establish quality for all recordings or full lip sequences.'],
            'training_ready':False,'human_transcript_review':'not_reviewed'}
    m.save(folder/'result.json',result)
    print(json.dumps({'result':m.bind(folder/'result.json'),'artifacts':artifacts,
                      'native_pixel_match':True,'roi_pixel_match':True},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
