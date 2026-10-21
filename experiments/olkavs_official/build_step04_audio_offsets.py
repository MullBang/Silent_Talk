"""Build a diagnostic time-mapping sidecar; does not change dataset timestamps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    measured_path = BASE/'results/step04_audio_alignment.json'
    measured_sha = '665701a2614b931ea9af9c8958ff5630382a301bbd7c81ef1e7f5b1acc4354ab'
    if sha(measured_path) != measured_sha:
        raise ValueError('Completed waveform evidence changed')
    measured = json.loads(measured_path.read_text(encoding='utf-8'))
    if measured['status'] != 'completed' or len(measured['sources']) != 4:
        raise ValueError('Need all four sources completed')
    sources = []
    for source in measured['sources']:
        anchors = source['anchors']
        offsets = [a['score']['offset_mp4_minus_original_seconds'] for a in anchors]
        correlations = [a['score']['absolute_correlation'] for a in anchors]
        ratios = [a['score']['peak_ratio'] for a in anchors]
        if len(anchors) != 3 or min(correlations)<.99 or max(offsets)-min(offsets)>=.02 or min(ratios)<1.5:
            raise ValueError('Independent source alignment did not meet strict candidate threshold')
        offset = round(statistics.median(offsets)*8000)/8000
        sources.append({
            'source_name':Path(source['video_path']).name,
            'source_stem':source['stem'], 'source_video_path':source['video_path'],
            'original_wav_path':source['original_wav_path'],
            'label_to_video_offset_seconds':offset,
            'offset_measurement': 'MP4 audio time minus original WAV time; rounded only to the measurement sample grid 1/8000s',
            'label_clock_assumption': 'JSON Audio_Name matches original WAV basename, declared WAV 48kHz and label duration agree with original WAV; full transcript content has not been independently checked',
            'formula': 'candidate_video_time = original_label_time + label_to_video_offset_seconds',
            'anchors_original_wav_seconds':[a['reference']['start_seconds'] for a in anchors],
            'anchors_offsets_seconds':offsets, 'anchors_absolute_correlations':correlations,
            'anchors_far_peak_ratios':ratios, 'offset_range_seconds':max(offsets)-min(offsets),
            'minimum_correlation':min(correlations),
            'original_wav_duration_seconds':source['original_wav_header']['duration_seconds'],
            'native_video_duration_seconds':source['video_native_duration_seconds'],
            'original_wav_minus_video_duration_seconds':source['original_wav_minus_video_duration_seconds'],
            'strict_waveform_gate_pass': True,
        })
    result = {
        'schema':'olkavs-step04-audio-offset-candidates-v1',
        'status':'validated',
        'validation_scope':'The four relative original-WAV to MP4-audio timeline mappings only; no certification of full transcript or audio-visual synchronization',
        'evidence_path':str(measured_path), 'evidence_sha256':measured_sha,
        'evidence_files':[{'path':str(measured_path),'sha256':measured_sha},
                          {'path':str(BASE/'results/step04_human_listening.json'),
                           'sha256':sha(BASE/'results/step04_human_listening.json')}],
        'builder_sha256':sha(Path(__file__)),
        'purpose':'Independent waveform-based candidate timestamp mapping for a separately identified diagnostic evaluation; original labels and prior results remain unchanged',
        'source_count':len(sources), 'sources':sources,
        'original_label_text_changed':False, 'original_label_timestamps_changed':False,
        'models_or_predictions_used_to_estimate_offsets':False,
        'limits':['Audio waveform correspondence does not independently prove every label text matches the spoken sentence.',
                  'Audio-to-video synchronization remains a separate issue; this sidecar maps to the MP4 audio timeline.',
                  'Three fixed 6-second anchors per recording support a constant offset but do not inspect every point in the full recording.',
                  'Do not apply these per-recording offsets to other files or training data without independent checks.'],
    }
    out = BASE/'results/step04_audio_offset_candidates.json'
    if out.exists():
        # Preserve the immediately preceding schema draft before adding the
        # caller's required status/evidence fields; numerical values are fixed.
        if sha(out) != 'cad00a0eb40b3c335d7ef4ad5dae7d974864a13371653a5c8f859467da9372f3':
            raise FileExistsError('Candidate sidecar already exists')
        draft = out.with_name('step04_audio_offset_candidates_schema_draft.json')
        if draft.exists():
            raise FileExistsError('Schema draft archive already exists')
        draft.write_bytes(out.read_bytes())
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(out),'sha256':sha(out),'offsets':{s['source_name']:s['label_to_video_offset_seconds'] for s in sources}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
