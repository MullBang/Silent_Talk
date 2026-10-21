"""Bounded independent waveform alignment of four original WAV/MP4 pairs.

Does not use lipreading output or optimize target text. Reads 3 fixed 6-second WAV
anchors and 36-second MP4 search windows per file, retaining small PCM excerpts.
No original label, video, WAV, crop or evaluation result is changed.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate

BASE = Path(__file__).resolve().parent
RATE = 8000
ANCHORS = (30.0, 150.0, 270.0)
ANCHOR_DURATION = 6.0
SEARCH_RADIUS = 15.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(executable: str, source: Path, start: float, duration: float, dest: Path, resume: bool) -> dict:
    command = [executable, '-nostdin', '-v', 'error', '-ss', f'{start:.9f}', '-i', str(source),
               '-t', f'{duration:.9f}', '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(RATE),
               '-c:a', 'pcm_s16le', '-n', str(dest)]
    reused = dest.exists()
    if reused and not resume:
        raise FileExistsError(dest)
    if not reused:
        process = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if process.returncode:
            raise RuntimeError(process.stderr)
    with wave.open(str(dest), 'rb') as stream:
        if stream.getframerate() != RATE or stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError('Unexpected decoded PCM format')
        count = stream.getnframes()
    if abs(count/RATE-duration) > 1/RATE:
        raise ValueError('Unexpected truncated excerpt')
    return {'source': str(source), 'start_seconds': start, 'requested_seconds': duration,
            'path': str(dest), 'sha256': sha(dest), 'bytes': dest.stat().st_size,
            'frames': count, 'sample_rate': RATE, 'channel_mix': 'ffmpeg mono downmix',
            'command': command, 'reused_completed_excerpt_from_interrupted_run': reused}


def read_pcm(path: str) -> np.ndarray:
    with wave.open(path, 'rb') as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float64)/32768


def correlate_windows(reference: np.ndarray, search: np.ndarray) -> dict:
    # Identical zero-phase passband on both signals; trimmed silence is never used.
    sos = butter(4, [100, 3400], btype='bandpass', fs=RATE, output='sos')
    reference = sosfiltfilt(sos, reference)
    search = sosfiltfilt(sos, search)
    reference = reference-reference.mean()
    n = len(reference)
    sums = np.concatenate(([0.], np.cumsum(search)))
    squares = np.concatenate(([0.], np.cumsum(search*search)))
    local_sum = sums[n:] - sums[:-n]
    local_square = squares[n:] - squares[:-n]
    centered_energy = np.maximum(local_square-local_sum**2/n, 0)
    ref_energy = float(np.dot(reference, reference))
    denominator = np.sqrt(ref_energy * centered_energy)
    numerator = correlate(search, reference, mode='valid', method='fft')
    normalized = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>1e-12)
    strength = np.abs(normalized)
    best = int(np.argmax(strength))
    alternative = strength.copy()
    exclusion = int(0.25*RATE)
    alternative[max(0,best-exclusion):min(len(alternative),best+exclusion+1)] = -1
    second = int(np.argmax(alternative))
    return {'best_start_sample_in_search': best, 'best_start_seconds_in_search': best/RATE,
            'signed_correlation': float(normalized[best]), 'absolute_correlation': float(strength[best]),
            'runner_up_exclusion_seconds': exclusion/RATE,
            'runner_up_start_seconds_in_search': second/RATE,
            'runner_up_absolute_correlation': float(strength[second]),
            'peak_margin': float(strength[best]-strength[second]),
            'peak_ratio': float(strength[best]/max(strength[second],1e-12)),
            'reference_rms_filtered': float(np.sqrt(ref_energy/n)),
            'matched_search_rms_filtered_centered': float(np.sqrt(centered_energy[best]/n))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-excerpts', action='store_true')
    args = parser.parse_args()
    output = BASE/'step04_audio_alignment'
    result_path = BASE/'results/step04_audio_alignment.json'
    previous = None
    if args.resume_excerpts:
        if not output.exists() or not result_path.exists():
            raise ValueError('Resume needs both prior output folder and checkpoint')
        saved = json.loads(result_path.read_text(encoding='utf-8'))
        if saved['status'] != 'running':
            raise ValueError('Do not resume completed run')
        backup = result_path.with_name('step04_audio_alignment_interrupted.json')
        if backup.exists():
            raise FileExistsError('Interrupted evidence already saved')
        backup.write_bytes(result_path.read_bytes())
        previous = {'path':str(backup), 'sha256':sha(backup),
                    'reason':'JSON serialization of numpy.bool_ in source summary interrupted run; existing PCM files are read and checked rather than regenerated. Numerical method unchanged.'}
    elif output.exists() or result_path.exists():
        raise FileExistsError('Refuse to replace alignment evidence')
    else:
        output.mkdir()
    sys.path.insert(0, str(BASE/'audio_tools'))
    import imageio_ffmpeg
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    inventory_path = BASE.parent.parent/'results/step18-intake/run01/inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    plan_path = BASE/'step03_source_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    native_path = BASE/'inputs_step03_recovery02/manifest.json'
    native = json.loads(native_path.read_text(encoding='utf-8'))
    result = {'schema': 'olkavs-step04-independent-audio-alignment-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'script_sha256': sha(Path(__file__)), 'inventory_sha256': sha(inventory_path),
              'resumed_interrupted_run':previous,
              'source_plan_sha256': sha(plan_path), 'prepared_manifest_sha256': sha(native_path),
              'ffmpeg': executable, 'ffmpeg_version': subprocess.check_output([executable,'-version'],text=True).splitlines()[0],
              'method': {'reference_clock': 'original WAV', 'search_clock': 'MP4 audio timeline',
                         'anchors_seconds': ANCHORS, 'anchor_duration_seconds': ANCHOR_DURATION,
                         'search_radius_seconds': SEARCH_RADIUS, 'sample_rate': RATE,
                         'filter': '4th-order Butterworth 100-3400Hz zero-phase on both signals',
                         'score': 'absolute Pearson-normalized cross-correlation, every 1/8000s shift',
                         'runner_up_exclusion_seconds': 0.25,
                         'offset_definition': 'MP4 matching start minus original WAV start',
                         'strong_anchor_rule': 'absolute correlation >=0.6 and far-peak ratio >=1.5',
                         'constant_offset_candidate_rule': 'all three anchors strong and offset range <=0.1s',
                         'candidate_is_not_label_fix': True},
              'audio_used_in_lipreading': False, 'new_asr': False, 'model_outputs_used': False,
              'original_sources_modified': False, 'full_original_source_copied': False,
              'user_listening_evidence': {
                  'source': 'user response conveyed by root agent in this task on 2026-09-24',
                  'excerpt_id': 'lip_J_1_F_02_C025_A_006_001',
                  'heard_text': '끝나는 거야',
                  'label_text': '아주 기쁘고 감사해',
                  'interpretation': 'User reports actual sampled speech disagrees with its assigned target; this alone does not establish whole-file offset or complete transcript truth.'},
              'sources': [], 'limits': [
                  'Waveform match verifies relative audio timelines, not transcript correctness or visual-audio synchronization.',
                  'All label corrections remain unapplied. Three anchors are a diagnostic for constant offset, not proof against any internal edits elsewhere.',
                  'One user-listened excerpt cannot independently certify every target sentence.',
                  'Reading streamed Google Drive source ranges may cause Drive caching beyond requested bytes.']}
    for source in plan['sources']:
        video = Path(source['source_video']['path'])
        matches = [r for partition in inventory['partitions'].values()
                   for r in partition['audio'] if Path(r['path']).stem==video.stem]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one inventoried matching original WAV: {video.stem}')
        wav_entry = matches[0]
        original_wav = Path(wav_entry['path'])
        for path, expected in ((video, source['source_video']), (original_wav, wav_entry)):
            stat = path.stat()
            if stat.st_size != expected['size_bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
                raise ValueError('Source metadata changed')
        with wave.open(str(original_wav),'rb') as stream:
            header = {'channels':stream.getnchannels(), 'sample_width_bytes':stream.getsampwidth(),
                      'sample_rate':stream.getframerate(), 'frame_count':stream.getnframes(),
                      'duration_seconds':stream.getnframes()/stream.getframerate(),
                      'compression':stream.getcomptype()}
        item = {'stem': video.stem, 'video_path':str(video), 'original_wav_path':str(original_wav),
                'original_wav_header':header, 'anchors':[]}
        native_source = native['sources'][video.name]
        item['video_native_duration_seconds'] = native_source['reported_frames']/native_source['native_fps']
        item['original_wav_minus_video_duration_seconds'] = header['duration_seconds']-item['video_native_duration_seconds']
        result['sources'].append(item)
        for anchor in ANCHORS:
            tag = f'{video.stem}_t{int(anchor):03}'
            reference = excerpt(executable,original_wav,anchor,ANCHOR_DURATION,output/f'{tag}_original_wav.wav',args.resume_excerpts)
            search_start = max(0,anchor-SEARCH_RADIUS)
            search = excerpt(executable,video,search_start,ANCHOR_DURATION+2*SEARCH_RADIUS,output/f'{tag}_mp4_search.wav',args.resume_excerpts)
            score = correlate_windows(read_pcm(reference['path']),read_pcm(search['path']))
            matched = search_start+score['best_start_seconds_in_search']
            score['matched_mp4_start_seconds'] = matched
            score['offset_mp4_minus_original_seconds'] = matched-anchor
            score['strong_anchor'] = score['absolute_correlation']>=0.6 and score['peak_ratio']>=1.5
            row = {'reference':reference,'mp4_search':search,'score':score}
            item['anchors'].append(row)
            result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'stem':video.stem,'anchor':anchor,'score':score},ensure_ascii=False),flush=True)
        offsets = np.array([r['score']['offset_mp4_minus_original_seconds'] for r in item['anchors']])
        slope,intercept = np.polyfit(np.array(ANCHORS),offsets,1)
        item['summary'] = {'median_offset_seconds':float(np.median(offsets)),
                           'offset_range_seconds':float(np.ptp(offsets)),
                           'linear_offset_intercept_seconds':float(intercept),
                           'linear_offset_slope_seconds_per_second':float(slope),
                           'linear_slope_ppm':float(slope*1e6),
                           'all_anchors_strong':all(r['score']['strong_anchor'] for r in item['anchors']),
                           'constant_offset_candidate':bool(all(r['score']['strong_anchor'] for r in item['anchors']) and np.ptp(offsets)<=0.1)}
    result['status'] = 'completed'
    result['extracted_original_wav_total_seconds'] = len(ANCHORS)*ANCHOR_DURATION*len(result['sources'])
    result['extracted_mp4_audio_total_seconds'] = len(ANCHORS)*(ANCHOR_DURATION+2*SEARCH_RADIUS)*len(result['sources'])
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'result':str(result_path),'sha256':sha(result_path),'sources':[{'stem':r['stem'],'summary':r['summary']} for r in result['sources']]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
