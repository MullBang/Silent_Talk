"""Prospective waveform mapping and review media for the frozen new120 only.

No model, ASR, training, G: read, original write, label edit, or prior59 offset.
Run ``freeze`` before ``run``. One bounded TAR recording is materialized at a
time using the immutable Step1 reader, then removed by its context manager.
Saved results are resumable only after their plan and artifact hashes validate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid
import wave

import cv2
import numpy as np
import PIL
import scipy

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE_PLAN = BASE / 'source_plan.json'
SOURCE_SHA = '5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929'
BUILDER = BASE / 'build_plan.py'
BUILDER_SHA = '71e42b20c66b7793d2c4a022d3a5f1892d38e235c56ca90fb96e00e5cd87256a'
REFINE = ROOT / 'experiments/training_alignment_step06/refine_alignment.py'
REFINE_SHA = '70a35c12507eb0ed851045133cfb137bc22d7bae00adcd7662ceac572790bd5a'
TIMELINE = ROOT / 'experiments/olkavs_official/inspect_timeline_step04.py'
TIMELINE_SHA = '5c2561f1895e67b7e452e777d49e23fccc40cd98338f78ce96191cc175c21a4a'
FFMPEG = ROOT / 'experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
PLAN = BASE / 'media_plan.json'
RESULTS = BASE / 'results'
RULES = {
    'global_anchors_seconds': [30., 150., 270.], 'global_reference_seconds': 6.,
    'global_search_radius_seconds': 15., 'global_search_seconds': 36.,
    'local_reference_seconds': 6., 'local_search_radius_seconds': 1., 'local_search_seconds': 8.,
    'pcm_rate': 8000, 'interpolation_factor': 6, 'evaluation_grid_rate': 48000,
    'minimum_correlation': .99, 'minimum_far_peak_ratio': 1.5,
    'minimum_filtered_rms': .0001, 'minimum_search_boundary_clearance_seconds': .5,
    'maximum_global_offset_range_exclusive_seconds': .02,
    'maximum_local_residual_inclusive_seconds': .001,
    'offset_formula': 'video_time = original_label_wav_time + offset',
    'recording_offset': 'median of three independent offsets rounded to 1/48000 seconds',
    'local_reference_start': '(original_start_time + original_end_time) / 2 - 3',
    'local_search_start': 'local_reference_start + recording_offset - 1',
    'out_of_bounds': 'quarantine; no clamping, adaptive anchors, inferred offsets or relaxed gates',
    'audio_output_rate': 16000, 'audio_channels': 1, 'audio_sample_width_bytes': 2,
    'context_radius_seconds': .5, 'student_fps': 25,
    'student_indices': 'round((original_start_time + recording_offset + k/25)*actual_video_fps), k < round((end-start)*25)',
    'roi_coordinate_order': 'yxyx -> xyxy; annotation indexed by native frame ordinal',
    'roi_margin_each_side': .2, 'roi_size': [96, 96],
    'roi_interpolation': 'cv2.INTER_LINEAR; uint8 BGR previews only',
    'roi_samples': ['first', 'middle', 'last'], 'preview_height': 360,
    'preview_codec': 'libx264', 'preview_preset': 'veryfast', 'preview_crf': 28,
    'preview_audio_codec': 'aac', 'preview_audio_kbps': 64,
    'pts_cfr_tolerance_seconds': .001, 'minimum_free_bytes': 10*1024**3,
    'audio_timeout_seconds': 120, 'preview_timeout_seconds': 180,
    'stop_after_consecutive_sources_with_zero_prepared_clips': 2,
}
LIMITS = [
    'The 1/48000-second evaluation grid interpolates saved 8 kHz PCM; it is not a new 48 kHz measurement.',
    'Audio waveform agreement supports a WAV-to-MP4 clock mapping, not transcript correctness or lip-audio synchronization.',
    'Three global and four local windows do not rule out every internal edit in a recording.',
    'A six-second local waveform window may include neighboring utterances; its correlation does not verify the exact selected transcript.',
    'First/middle/last ROI and CFR PTS are visual samples, not a full sequence ROI or production parity validation.',
    'All 120 original labels and train96/val24 readiness roles are preserved. No row is approved for training.',
    'Human transcript listening and visual review have not been performed by this program.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), f'Workspace evidence required: {path}')
    return path


def sha(path):
    with local(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    return {'path': str(local(path)), 'sha256': sha(path)}


def read(path):
    return json.loads(local(path).read_text(encoding='utf-8-sig'))


def save(path, value, replace=False):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(data, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(data)


def module(name, path, expected):
    require(sha(path) == expected, f'Frozen helper changed: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def helpers():
    return (module('new120_source_contract', BUILDER, BUILDER_SHA),
            module('new120_immutable_waveform_estimator', REFINE, REFINE_SHA),
            module('new120_immutable_mp4_metadata', TIMELINE, TIMELINE_SHA))


def environment():
    return {'python': sys.version, 'executable': bind(sys.executable),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'opencv': cv2.__version__, 'pillow': PIL.__version__}


def freeze():
    builder, _, _ = helpers()
    sources = builder.load_plan(SOURCE_PLAN, SOURCE_SHA)
    require(np.__version__ == '1.26.4' and scipy.__version__ == '1.17.1', 'Use frozen .venv-olkavs-roi runtime')
    require(Path(sys.executable).resolve() == (ROOT/'.venv-olkavs-roi/Scripts/python.exe').resolve(), 'Wrong Python executable')
    command = [str(FFMPEG), '-version']
    version = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()[0]
    code = [Path(__file__), BASE/'test_prepare_media.py', BUILDER, REFINE, TIMELINE,
            builder.TAR_SOURCE, builder.TOKENIZER]
    plan = {'schema': 'lipnet-new120-media-plan-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'frozen_before_first_new_media_extraction': True, 'source_plan': bind(SOURCE_PLAN),
            'source_order': [s['stem'] for s in sources['sources']],
            'ids': [r['id'] for r in sources['records']], 'code': [bind(p) for p in code],
            'ffmpeg': bind(FFMPEG), 'ffmpeg_version': version, 'environment': environment(),
            'rules': RULES, 'limits': LIMITS,
            'scope': {'G_access': False, 'original_writes': False, 'label_edits': False,
                      'old59_offsets_used': False, 'model_execution': False, 'ASR': False,
                      'training': False, 'training_ready': False, 'human_transcript_review': 'not_reviewed'}}
    require(not (RESULTS/'sources').exists() and not (RESULTS/'media').exists(), 'Media output already exists before freeze')
    save(PLAN, plan)
    print(json.dumps({'media_plan': bind(PLAN), 'sources': 30, 'clips': 120}, ensure_ascii=False), flush=True)


def verify_plan(expected_sha256):
    require(isinstance(expected_sha256,str) and sha(PLAN)==expected_sha256, 'Explicit frozen media-plan SHA is required and must match')
    plan = read(PLAN)
    require(plan['schema'] == 'lipnet-new120-media-plan-v1' and plan['rules'] == RULES and plan['limits'] == LIMITS, 'Media rules changed')
    require(plan['source_plan'] == bind(SOURCE_PLAN) and plan['source_plan']['sha256'] == SOURCE_SHA, 'Source plan changed')
    for item in [*plan['code'], plan['ffmpeg']]:
        require(bind(item['path']) == item, 'Frozen media execution dependency changed: ' + item['path'])
    require(environment() == plan['environment'], 'Runtime differs from media freeze')
    builder, scoring, timeline = helpers()
    sources = builder.load_plan(SOURCE_PLAN, SOURCE_SHA)
    require(plan['source_order'] == [s['stem'] for s in sources['sources']] and plan['ids'] == [r['id'] for r in sources['records']], 'Frozen source/clip order changed')
    return plan, sources, builder, scoring, timeline


def in_bounds(start, seconds, duration):
    require(all(math.isfinite(x) for x in (start, seconds, duration)) and seconds > 0 and start >= 0
            and start + seconds <= duration + 1e-9, 'Requested window outside source; no clamping allowed')


def wav_info(path):
    with wave.open(str(local(path)), 'rb') as stream:
        return {'sample_rate': stream.getframerate(), 'channels': stream.getnchannels(),
                'sample_width_bytes': stream.getsampwidth(), 'frames': stream.getnframes(),
                'compression': stream.getcomptype(), 'duration_seconds': stream.getnframes()/stream.getframerate()}


def excerpt(source, member, role, start, seconds, duration, output, rate, ffmpeg):
    in_bounds(start, seconds, duration)
    require(not output.exists(), 'Refuse to overwrite saved PCM')
    # Input seeking with transcoding performs accurate seek; exact PCM length is required below.
    command = [ffmpeg, '-nostdin', '-v', 'error', '-ss', f'{start:.9f}', '-i', str(source),
               '-t', f'{seconds:.9f}', '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(rate),
               '-c:a', 'pcm_s16le', '-n', str(output)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=RULES['audio_timeout_seconds'])
    require(proc.returncode == 0, 'FFmpeg PCM failed: ' + proc.stderr[-2000:])
    info = wav_info(output)
    require((info['sample_rate'], info['channels'], info['sample_width_bytes'], info['compression']) == (rate, 1, 2, 'NONE'), 'Wrong PCM encoding')
    require(info['frames'] == round(seconds*rate), 'Incomplete PCM excerpt')
    return {**bind(output), **info, 'source_role': role, 'source_sha256': member['sha256'],
            'clock': 'original_label_wav' if role == 'audio' else 'mp4',
            'start_seconds': start, 'end_seconds': start+seconds, 'requested_seconds': seconds,
            'bytes': output.stat().st_size, 'command': command}


def pcm(spec):
    require(sha(spec['path']) == spec['sha256'], 'PCM changed before scoring')
    with wave.open(spec['path'], 'rb') as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) == (8000, 1, 2, 'NONE'), 'Score expects mono8k PCM16')
        require(stream.getnframes() == spec['frames'], 'PCM frame count changed')
        value = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float64)/32768
    require(value.size == spec['frames'] and np.isfinite(value).all(), 'Invalid PCM payload')
    return value


def score_pair(reference, search, scoring):
    score = scoring.global_score(pcm(reference), pcm(search))
    score['offset_mp4_minus_original_seconds'] = search['start_seconds'] + score['best_start_seconds_in_search'] - reference['start_seconds']
    return score


def offset_summary(anchors):
    require(len(anchors) == 3, 'Need exactly three global anchors')
    offsets = [a['score']['offset_mp4_minus_original_seconds'] for a in anchors]
    require(all(math.isfinite(o) for o in offsets), 'Nonfinite offset')
    median = round(statistics.median(offsets)*48000)/48000
    spread = max(offsets)-min(offsets)
    return {'median_offset_seconds': median, 'offset_range_seconds': spread,
            'all_anchors_strict': all(a['score']['strict_anchor_pass'] for a in anchors),
            'constant_mapping_supported': all(a['score']['strict_anchor_pass'] for a in anchors) and spread < .02}


def local_check_gate(score, source_offset):
    residual = score['offset_mp4_minus_original_seconds'] - source_offset
    # No rounding or tolerance that could admit a value above the prospective gate.
    return {'offset_residual_seconds': residual, 'maximum_absolute_residual_seconds': .001,
            'residual_gate_pass': abs(residual) <= .001,
            'verified': score['strict_anchor_pass'] and abs(residual) <= .001}


def target_indices(record, offset, fps, frame_count, video_duration):
    start, end = record['start_time']+offset, record['end_time']+offset
    in_bounds(start, end-start, video_duration)
    n = round((record['end_time']-record['start_time'])*25)
    require(n == record['expected_frames'] and n >= record['ctc_min_frames'] > 0, 'Student length/CTC feasibility differs')
    times = [start+k/25 for k in range(n)]
    indices = [round(t*fps) for t in times]
    require(indices[0] >= 0 and indices[-1] < frame_count and all(a < b for a,b in zip(indices,indices[1:])), 'Native frame range or order invalid')
    return {'corrected_start_time': start, 'corrected_end_time': end, 'student_frames': n,
            'ctc_min_frames': record['ctc_min_frames'], 'ctc_feasible': True,
            'actual_video_fps': fps, 'native_frame_indices': indices, 'target_times_seconds': times}


def boxes_yxyx(doc, key, count, width, height):
    boxes = np.asarray(doc['Bounding_box_info'][key]['xtl_ytl_xbr_ybr'], dtype=np.float64)
    require(boxes.shape == (count,4) and np.isfinite(boxes).all(), 'Annotation count/shape/finite mismatch')
    xy = boxes[:, [1,0,3,2]]
    require(bool(np.all((xy[:,0] >= 0)&(xy[:,1] >= 0)&(xy[:,0] < xy[:,2])&(xy[:,1] < xy[:,3])
                       &(xy[:,2] <= width)&(xy[:,3] <= height))), 'Annotation geometry outside frame after yxyx conversion')
    return boxes


def validate_label(doc, source, records, tokenizer):
    require(isinstance(doc,list) and len(doc) == 1 and isinstance(doc[0],dict), 'Unexpected original label schema')
    doc = doc[0]
    require(Path(doc['Video_info']['video_Name']).name == source['stem']+'.mp4'
            and Path(doc['Audio_info']['Audio_Name']).name == source['stem']+'.wav', 'Label/media identity differs')
    require(doc['speaker_info']['speaker_ID'] == source['speaker_id'] and doc['Video_env']['Angle'] == 'A', 'Speaker/angle mismatch')
    annotated = {int(r['ID']): r for r in doc['Sentence_info']}
    require(len(annotated) == len(doc['Sentence_info']), 'Duplicate sentence label ID')
    require(float(doc['Video_info']['FPS']) == source['label_fps'], 'Nominal label FPS changed')
    for row in records:
        original = annotated[row['sentence_id']]
        require(original['sentence_text'] == row['source_text'] and original['start_time'] == row['start_time']
                and original['end_time'] == row['end_time'], 'Original sentence text/time differs from frozen selection')
        tokens = tokenizer.encode_text(row['source_text'])
        require(tokenizer.normalize_text(row['source_text']) == row['text'] and len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:])) == row['ctc_min_frames'], 'Token/CTC label contract changed')
    return doc


def duration_bound(metadata, handler):
    tracks = [t for t in metadata['tracks'] if t['handler'] == handler]
    require(len(tracks) == 1, 'Require one MP4 track for ' + handler)
    track = tracks[0]
    durations = [metadata['movie_clock']['duration_seconds'], track['media_clock']['duration_seconds']]
    if track.get('edit_list'):
        require(all(e['media_rate_integer'] == 1 and e['media_rate_fraction'] == 0 for e in track['edit_list']), 'Non-unit MP4 edit rate')
        durations.append(sum(e['movie_duration_seconds'] for e in track['edit_list']))
    result = min(durations)
    require(math.isfinite(result) and result > 0, 'Invalid MP4 track duration')
    return result


def preview_crop(frame, box_yxyx):
    """Small preview implementation of the documented margin; no production import."""
    h,w = frame.shape[:2]
    y1,x1,y2,x2 = np.asarray(box_yxyx,dtype=np.float32).tolist()
    require(0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, 'Invalid preview crop box')
    bw,bh = x2-x1,y2-y1
    rectangle = [max(0,min(math.floor(x1-.2*bw),w-1)), max(0,min(math.floor(y1-.2*bh),h-1)),
                 min(w,math.ceil(x2+.2*bw)), min(h,math.ceil(y2+.2*bh))]
    a,b,c,d = rectangle
    return cv2.resize(frame[b:d,a:c],(96,96),interpolation=cv2.INTER_LINEAR),rectangle


def array_binding(value):
    return {'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype), 'color_order': 'BGR'}


def write_png(path, array):
    require(not path.exists(), 'Refuse to overwrite PNG evidence')
    ok, buffer = cv2.imencode('.png',array)
    require(ok, 'PNG encoding failed')
    with path.open('xb') as stream:
        stream.write(buffer.tobytes())
    return {**bind(path), 'pixels': array_binding(array)}


def roi_previews(capture, source, record, targets, lips, faces, folder):
    positions = [0,targets['student_frames']//2,targets['student_frames']-1]
    samples, context = [], None
    for name,k in zip(RULES['roi_samples'],positions):
        index = targets['native_frame_indices'][k]
        require(capture.set(cv2.CAP_PROP_POS_FRAMES,index), 'Frame seek failed')
        ok,frame = capture.read()
        require(ok and frame is not None and frame.dtype == np.uint8, 'Preview frame decode failed')
        pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
        index_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        fps = targets['actual_video_fps']
        require(math.isfinite(pts) and abs(index_after-index-1) <= 1e-6 and abs(pts-index/fps) <= .001
                and abs(pts-targets['target_times_seconds'][k]) <= .5/fps+1e-6, 'Preview frame PTS/index mismatch')
        require(frame.shape[:2] == (source['video_metadata']['height'],source['video_metadata']['width']), 'Decoded frame dimensions differ')
        roi,rectangle = preview_crop(frame,lips[index])
        entry = {'position': name,'student_index': k,'native_frame_index': index,'decoded_pts_seconds': pts,
                 'target_time_seconds': targets['target_times_seconds'][k], 'annotation_box_index': index,
                 'bbox_yxyx': lips[index].tolist(),'bbox_xyxy': lips[index][[1,0,3,2]].tolist(),
                 'crop_xyxy': rectangle,'raw_frame_array': array_binding(frame),
                 'roi_array': array_binding(roi),'image': write_png(folder/(name+'_lip.png'),roi)}
        samples.append(entry)
        if name == 'middle':
            context_pixels = cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
            context = {'native_frame_index': index,'decoded_pts_seconds':pts,
                       'raw_frame_array':array_binding(frame),'face_bbox_yxyx':faces[index].tolist(),
                       'face_bbox_xyxy':faces[index][[1,0,3,2]].tolist(),
                       'kind':'whole original frame resized to480x270; no face crop or overlay',
                       'image':write_png(folder/'middle_context.png',context_pixels)}
    return samples,context


def video_preview(video, source, start, seconds, duration, output, ffmpeg):
    in_bounds(start,seconds,duration)
    require(not output.exists(), 'Refuse preview overwrite')
    command = [ffmpeg,'-nostdin','-v','error','-ss',f'{start:.9f}','-i',str(video),'-t',f'{seconds:.9f}',
               '-map','0:v:0','-map','0:a:0','-vf','scale=-2:360','-c:v','libx264','-preset','veryfast',
               '-crf','28','-pix_fmt','yuv420p','-c:a','aac','-b:a','64k','-ar','16000','-ac','1',
               '-movflags','+faststart','-n',str(output)]
    process = subprocess.run(command,capture_output=True,text=True,timeout=RULES['preview_timeout_seconds'])
    require(process.returncode == 0 and output.stat().st_size > 0, 'Review MP4 encode failed: '+process.stderr[-2000:])
    return {**bind(output),'source_role':'video','source_sha256':source['members']['video']['sha256'],
            'clock':'mp4','start_seconds':start,'end_seconds':start+seconds,'requested_seconds':seconds,
            'review_only':True,'command':command}


def initial_clip(row, source):
    return {'id':row['id'],'split':row['split'],'speaker_id':row['speaker_id'],'source_stem':source['stem'],
            'reference':row['source_text'],'text':row['text'],'sentence_id':row['sentence_id'],
            'start_time':row['start_time'],'end_time':row['end_time'],
            'label_sha256':row['label_sha256'],'status':'quarantined','timing_status':'not_verified',
            'artifacts_ready':False,'audio':None,'context_audio':None,'mp4_audio':None,'preview_video':None,
            'roi_samples':[],'face_context':None,'source_contact_sheet':None,'local_check':None,
            'training_ready':False,'human_transcript_review':'not_reviewed','human_visual_review':'not_reviewed'}


def invalidate_source_clips(clips, reason):
    """A late archive integrity/cleanup error withdraws earlier timing approvals."""
    for clip in clips:
        clip['status']='quarantined'; clip['artifacts_ready']=False
        clip['timing_status']='invalidated_source_integrity'
        clip['quarantine_reason']=reason
        if clip['local_check'] is not None:
            clip['local_check']['waveform_gate_pass_before_source_invalidation']=clip['local_check']['verified']
            clip['local_check']['verified']=False
            clip['local_check']['invalidated_by_source_error']=True


def contact_sheet(clips, path):
    sheet = np.full((len(clips)*210,720,3),245,dtype=np.uint8)
    for row,clip in enumerate(clips):
        y = row*210
        cv2.putText(sheet,clip['id'],(8,y+19),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
        if clip['face_context']:
            image = cv2.imread(clip['face_context']['image']['path'])
            require(image is not None, 'Context image read failed')
            sheet[y+25:y+205,8:328] = cv2.resize(image,(320,180),interpolation=cv2.INTER_AREA)
        for column,sample in enumerate(clip['roi_samples']):
            image = cv2.imread(sample['image']['path'])
            require(image is not None and image.shape == (96,96,3), 'ROI image read failed')
            x=348+column*120
            cv2.putText(sheet,sample['position'],(x,y+52),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
            sheet[y+62:y+158,x:x+96]=image
        if clip['status'] != 'prepared':
            cv2.putText(sheet,'QUARANTINED - see JSON',(340,y+186),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,180),1,cv2.LINE_AA)
    return write_png(path,sheet)


def artifact_list(folder):
    return [bind(p) for p in sorted(folder.rglob('*')) if p.is_file()]


def prepare_source(source, rows, plan_binding, plan, builder, scoring, timeline):
    started=time.perf_counter()
    folder=RESULTS/'media'/source['stem']/uuid.uuid4().hex
    folder.mkdir(parents=True)
    item={'schema':'lipnet-new120-media-source-v1','source_stem':source['stem'],'speaker_id':source['speaker_id'],
          'split':source['split'],'source_plan':bind(SOURCE_PLAN),'media_plan':plan_binding,
          'status':'quarantined','anchors':[],'offset_summary':None,'validation':{},
          'clips':[initial_clip(r,source) for r in rows], 'training_ready':False,
          'human_transcript_review':'not_reviewed','artifact_directory':str(folder)}
    try:
        with builder.transport.materialize_source(source,BASE/'temporary',minimum_free_bytes=RULES['minimum_free_bytes']) as paths:
            require(sha(paths['label']) == source['label_copy']['sha256'] == source['members']['label']['sha256'], 'Original/cached label SHA mismatch')
            require(bind(source['label_copy']['path']) == source['label_copy'], 'Cached label changed')
            label=validate_label(read(paths['label']),source,rows,builder.tokenizer)
            metadata=timeline.inspect(paths['video'])
            mp4_audio_duration=duration_bound(metadata,'soun')
            video_duration=min(duration_bound(metadata,'vide'),source['video_duration_seconds'])
            wav=wav_info(paths['audio'])
            require(wav['sample_rate']==48000 and wav['compression']=='NONE' and wav['channels']==2 and wav['sample_width_bytes']==2, 'Unexpected original WAV format')
            capture=cv2.VideoCapture(str(paths['video']))
            try:
                require(capture.isOpened(), 'OpenCV cannot open materialized video')
                fps=float(capture.get(cv2.CAP_PROP_FPS)); count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                require(math.isfinite(fps) and abs(fps-source['video_fps']) <= 1e-9 and count==source['video_frame_count']
                        and width==source['video_metadata']['width'] and height==source['video_metadata']['height'], 'Actual video metadata differs from full-decode audit')
                lips=boxes_yxyx(label,'Lip_bounding_box',count,width,height)
                faces=boxes_yxyx(label,'Face_bounding_box',count,width,height)
                item['validation']={'original_members_payload_sha256_verified':True,'original_and_cached_label_exact':True,
                    'selected_sentences_verified':len(rows),'all_lip_and_face_boxes_valid':True,'bbox_count':count,
                    'actual_video_fps':fps,'video_frame_count':count,'width':width,'height':height,
                    'video_duration_bound_seconds':video_duration,'mp4_audio_duration_bound_seconds':mp4_audio_duration,
                    'original_wav_header':wav,'mp4_metadata':metadata,'old59_offsets_used':False}
                ffmpeg=plan['ffmpeg']['path']
                for anchor in RULES['global_anchors_seconds']:
                    reference=excerpt(paths['audio'],source['members']['audio'],'audio',anchor,6.,wav['duration_seconds'],folder/f'anchor_{int(anchor):03}_reference.wav',8000,ffmpeg)
                    search=excerpt(paths['video'],source['members']['video'],'video',anchor-15,36.,mp4_audio_duration,folder/f'anchor_{int(anchor):03}_search.wav',8000,ffmpeg)
                    item['anchors'].append({'reference':reference,'mp4_search':search,'score':score_pair(reference,search,scoring)})
                item['offset_summary']=offset_summary(item['anchors'])
                require(item['offset_summary']['constant_mapping_supported'], 'Strict global waveform mapping gate failed')
                offset=item['offset_summary']['median_offset_seconds']
                for row,clip in zip(rows,item['clips']):
                    clip_folder=folder/row['id']; clip_folder.mkdir()
                    try:
                        start,end=row['start_time'],row['end_time']; duration=end-start
                        ref_start=(start+end)/2-3
                        search_start=ref_start+offset-1
                        reference=excerpt(paths['audio'],source['members']['audio'],'audio',ref_start,6.,wav['duration_seconds'],clip_folder/'local_reference.wav',8000,ffmpeg)
                        search=excerpt(paths['video'],source['members']['video'],'video',search_start,8.,mp4_audio_duration,clip_folder/'local_search.wav',8000,ffmpeg)
                        score=score_pair(reference,search,scoring)
                        clip['local_check']={'reference':reference,'mp4_search':search,'score':score,
                            'source_offset_seconds':offset,**local_check_gate(score,offset)}
                        require(clip['local_check']['verified'], 'Strict local waveform/residual gate failed')
                        targets=target_indices(row,offset,fps,count,video_duration)
                        clip['timing']=targets; clip['timing_status']='verified'
                        clip['audio']=excerpt(paths['audio'],source['members']['audio'],'audio',start,duration,wav['duration_seconds'],clip_folder/'label_audio.wav',16000,ffmpeg)
                        clip['context_audio']=excerpt(paths['audio'],source['members']['audio'],'audio',start-.5,duration+1.,wav['duration_seconds'],clip_folder/'context_audio.wav',16000,ffmpeg)
                        clip['mp4_audio']=excerpt(paths['video'],source['members']['video'],'video',start+offset,duration,mp4_audio_duration,clip_folder/'mp4_audio.wav',16000,ffmpeg)
                        clip['roi_samples'],clip['face_context']=roi_previews(capture,source,row,targets,lips,faces,clip_folder)
                        # Optional encoded preview failure is reported; core PCM+ROI readiness is independent.
                        try:
                            clip['preview_video']=video_preview(paths['video'],source,start+offset,duration,min(video_duration,mp4_audio_duration),clip_folder/'preview.mp4',ffmpeg)
                        except (ValueError,OSError,subprocess.SubprocessError) as exc:
                            clip['preview_error']=str(exc)
                        clip['status']='prepared'; clip['artifacts_ready']=True
                    except (ValueError,OSError,subprocess.SubprocessError) as exc:
                        clip['quarantine_reason']=str(exc)
                item['status']='prepared' if all(c['status']=='prepared' for c in item['clips']) else 'partial_quarantine'
            finally:
                capture.release()
    except Exception as exc:
        item['status']='quarantined'
        item['quarantine_reason']=f'{type(exc).__name__}: {exc}'
        # A source integrity error can invalidate earlier successful clips too.
        invalidate_source_clips(item['clips'],item['quarantine_reason'])
    sheet=contact_sheet(item['clips'],folder/'contact_sheet.png')
    item['contact_sheet']=sheet
    for clip in item['clips']:
        clip['source_contact_sheet']=sheet
    item['artifacts_all']=artifact_list(folder)
    item['elapsed_seconds']=time.perf_counter()-started
    item['temporary_originals_cleaned']=not any((BASE/'temporary').glob(source['stem']+'-*'))
    return item


def verify_saved_result(item, source, rows, plan_binding):
    require(item['schema']=='lipnet-new120-media-source-v1' and item['media_plan']==plan_binding
            and item['source_plan']==bind(SOURCE_PLAN) and item['source_stem']==source['stem'], 'Wrong resumable source result binding')
    require(item['speaker_id']==source['speaker_id'] and item['split']==source['split']
            and item['training_ready'] is False and item['human_transcript_review']=='not_reviewed', 'Resumed source role/review differs')
    require([c['id'] for c in item['clips']]==[r['id'] for r in rows], 'Resumed source omitted/reordered rows')
    for clip,row in zip(item['clips'],rows):
        require(clip['reference']==row['source_text'] and clip['text']==row['text'] and clip['split']==row['split']
                and clip['speaker_id']==row['speaker_id'] and clip['start_time']==row['start_time']
                and clip['end_time']==row['end_time'] and clip['training_ready'] is False
                and clip['human_transcript_review']=='not_reviewed', 'Resumed row changed labels/roles/review status')
    folder=local(item['artifact_directory'])
    require(folder.is_relative_to((RESULTS/'media'/source['stem']).resolve()), 'Unexpected artifact directory')
    require(item['artifacts_all']==artifact_list(folder), 'Saved artifact missing, added, or changed')
    if item['status'] in ('prepared','partial_quarantine'):
        require(item['offset_summary']==offset_summary(item['anchors']) and item['offset_summary']['constant_mapping_supported'], 'Resumed global mapping differs')
    else:
        require(all(c['timing_status']!='verified' and not c['artifacts_ready'] for c in item['clips']), 'Quarantined source retains approved clips')
    for clip in item['clips']:
        for key in ['audio','context_audio','mp4_audio']:
            spec=clip[key]
            if spec is not None:
                require(local(spec['path']).is_relative_to(folder) and bind(spec['path'])=={k:spec[k] for k in ['path','sha256']}, 'Resumed PCM reference changed')
                info=wav_info(spec['path'])
                require(all(info[k]==spec[k] for k in info) and (info['sample_rate'],info['channels'],info['sample_width_bytes'])==(16000,1,2), 'Resumed PCM metadata differs')
        if clip['artifacts_ready']:
            require(clip['status']=='prepared' and clip['timing_status']=='verified' and clip['local_check']['verified']
                    and all(clip[k] is not None for k in ['audio','context_audio','mp4_audio','face_context'])
                    and len(clip['roi_samples'])==3, 'Resumed prepared clip lacks required evidence')
            gate=local_check_gate(clip['local_check']['score'],item['offset_summary']['median_offset_seconds'])
            require(all(clip['local_check'][k]==v for k,v in gate.items()) and gate['verified'], 'Resumed local mapping differs')


def run(expected_sha256):
    plan,sources,builder,scoring,timeline=verify_plan(expected_sha256)
    plan_binding=bind(PLAN)
    RESULTS.mkdir(exist_ok=True)
    lock=RESULTS/'media_run.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(datetime.now(timezone.utc).isoformat())
    try:
        completed=[]
        empty_sources_in_a_row=0
        progress_path=RESULTS/'media_progress.json'
        progress=read(progress_path) if progress_path.exists() else {'media_plan':plan_binding,'sources':[]}
        require(progress['media_plan']==plan_binding, 'Progress belongs to a different frozen plan')
        refs={r['path']:r for r in progress['sources']}
        require(len(refs)==len(progress['sources']), 'Duplicate progress source')
        allowed_paths={str((RESULTS/'sources'/(s['stem']+'.json')).resolve()) for s in sources['sources']}
        require(set(refs).issubset(allowed_paths) and all(Path(p).is_file() for p in refs), 'Progress contains unknown or missing source results')
        for source in sources['sources']:
            rows=[r for r in sources['records'] if r['id'] in source['record_ids']]
            target=RESULTS/'sources'/(source['stem']+'.json')
            if target.exists():
                require(refs.get(str(target.resolve()))==bind(target), 'Existing result is changed or uncommitted in progress; manual review required')
                item=read(target)
                verify_saved_result(item,source,rows,plan_binding)
            else:
                item=prepare_source(source,rows,plan_binding,plan,builder,scoring,timeline)
                save(target,item)
                verify_saved_result(item,source,rows,plan_binding)
            completed.append((bind(target),item))
            progress={'media_plan':plan_binding,'sources':[ref for ref,_ in completed]}
            # Include later saved records on a resumed run until their turn is validated.
            progress['sources'] += [ref for path,ref in refs.items() if path not in {r['path'] for r in progress['sources']}]
            save(progress_path,progress,replace=True)
            print(json.dumps({'source':source['stem'],'status':item['status'],'completed_sources':len(completed),
                              'clips_prepared':sum(c['artifacts_ready'] for c in item['clips']),
                              'offset':item['offset_summary']},ensure_ascii=False),flush=True)
            empty_sources_in_a_row=0 if any(c['artifacts_ready'] for c in item['clips']) else empty_sources_in_a_row+1
            require(empty_sources_in_a_row<RULES['stop_after_consecutive_sources_with_zero_prepared_clips'],
                    'Two consecutive sources yielded no prepared clips; stopped for diagnosis with completed evidence preserved')
        by_id={c['id']:c for _,item in completed for c in item['clips']}
        require(len(by_id)==120 and set(by_id)==set(plan['ids']), 'Final manifest lost selected IDs')
        clips=[by_id[r['id']] for r in sources['records']]
        verify_plan(expected_sha256)
        require(bind(PLAN)==plan_binding, 'Media plan changed during execution')
        manifest={'schema':'lipnet-new120-media-manifest-v1','status':'completed_all_planned_sources',
                  'source_plan':bind(SOURCE_PLAN),'media_plan':plan_binding,'sources':[ref for ref,_ in completed],
                  'clips':clips,'counts':{'sources':30,'clips':120,'train':96,'val':24,
                      'artifacts_ready':sum(c['artifacts_ready'] for c in clips),
                      'timing_verified':sum(c['timing_status']=='verified' for c in clips),
                      'status':dict(Counter(c['status'] for c in clips))},
                  'training_ready':False,'human_transcript_review':'not_reviewed','model_execution':False,
                  'temporary_originals_cleaned':all(item['temporary_originals_cleaned'] for _,item in completed),'limits':LIMITS}
        path=RESULTS/'media_manifest.json'
        if path.exists():
            require(read(path)==manifest, 'Existing final manifest differs; no overwrite')
        else:
            save(path,manifest)
        print(json.dumps({'manifest':bind(path),'counts':manifest['counts']},ensure_ascii=False),flush=True)
    finally:
        lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','verify-plan'])
    parser.add_argument('--expected-plan-sha256')
    args=parser.parse_args()
    if args.action=='freeze':
        freeze()
    elif args.action=='run':
        run(args.expected_plan_sha256)
    else:
        verify_plan(args.expected_plan_sha256); print(json.dumps({'plan':bind(PLAN),'verified':True}),flush=True)


if __name__=='__main__':
    main()
