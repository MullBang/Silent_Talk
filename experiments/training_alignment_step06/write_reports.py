"""Summarize independently audited stage06 evidence without reading cloud media."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'results'
DOCS = ROOT / 'docs/공식모델_도입검증'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Only local workspace artifacts may be read')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def save_once(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(path.read_text(encoding='utf-8') == encoded, f'Existing output differs: {path}')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(encoded)


def main():
    inventory = read(BASE / 'training_inventory.json')
    plan = read(BASE / 'plan.json')
    refinement = read(BASE / 'refinement_plan.json')
    original_progress = read(OUT / 'progress.json')
    fine_progress = read(OUT / 'refinement_progress.json')
    original_audit = read(OUT / 'independent_audit.json')
    fine_audit = read(OUT / 'independent_refined_audit.json')
    inventory_sha = sha(BASE/'training_inventory.json')
    require(inventory_sha == plan['inventory']['sha256'] == refinement['inventory']['sha256']
            == original_audit['inventory_sha256'], 'Current inventory differs from audited inventory')
    for progress in (original_progress, fine_progress):
        require(progress['completed_sources'] == 59, 'All59 sources required')
        for spec in progress['source_results']:
            require(sha(spec['path']) == spec['sha256'], 'Source result changed')
    for audit in (original_audit, fine_audit):
        require(audit['status'] == 'passed', 'Independent audit must pass')
        require(audit['verified']['source_outcomes'] == 59, 'Independent coverage incomplete')
    require(original_audit['plan_sha256'] == sha(BASE/'plan.json'), 'Original audit plan differs')
    require(original_audit['progress_snapshot_sha256'] == sha(OUT/'progress.json'), 'Original audit progress differs')
    require(fine_audit['refinement_plan_sha256'] == sha(BASE/'refinement_plan.json'), 'Refined audit plan differs')
    require(fine_audit['refinement_progress_sha256'] == sha(OUT/'refinement_progress.json'), 'Refined audit progress differs')
    require(fine_audit['original_independent_audit_sha256'] == sha(OUT/'independent_audit.json'), 'Original audit binding differs')

    sources = {s['source_stem']: s for s in inventory['sources']}
    originals = {s: read(OUT/f'sources/{s}.json') for s in plan['source_order']}
    refined = {s: read(OUT/f'refined_sources/{s}.json') for s in plan['source_order']}
    supported = [r for r in refined.values() if r['status'] == 'supported']
    quarantined = [r for r in refined.values() if r['status'] != 'supported']
    offsets = Counter(r['summary']['median_offset_seconds'] for r in supported)
    anchors = [a for r in refined.values() for a in r['anchors']]
    require(len(anchors) == fine_audit['verified']['independent_full_search_anchors'] == 177,
            'This final report requires all177 refined anchor outcomes')
    require(original_audit['verified']['PCM_hash_format_length_and_command_count'] ==
            fine_audit['verified']['unique_PCM_hash_format_length_command_checks'] == 354,
            'This final report requires all354 independently checked PCM excerpts')
    require(all(v.get('summary') is not None and v.get('reason') in
                (None, 'At least one strict anchor gate or constant-offset range failed') for v in originals.values()),
            'Source-integrity failures require a separate incomplete-evidence report')
    available = read(OUT/'availability.json')['sources']
    require(len(available) == 59 and all(f['status'] == 'available' for s in available for f in s['files'].values()),
            'All59 original sets must have been available')
    require(supported, 'No supported mapping: write a dedicated inconclusive report instead')
    impacts = {v['id']: v for r in supported for v in r['training_frame_selection_impact']}
    record_lookup = {r['id']: (r, src['source_stem']) for src in sources.values() for r in src['training_records']}
    branches = {}
    for branch, ids in inventory['branch_sample_ids'].items():
        selected = [impacts[i] for i in ids if i in impacts]
        branches[branch] = {
            'clips': len(ids), 'supported_mapping_clips': len(selected),
            'unresolved_clips': len(ids)-len(selected),
            'target_frames': sum(v['target_frames'] for v in selected),
            'clips_with_changed_indices': sum(v['changed_selected_frame_indices'] > 0 for v in selected),
            'changed_frame_indices': sum(v['changed_selected_frame_indices'] for v in selected),
            'out_of_bounds_ids': [v['id'] for v in selected if not v['fully_within_video_duration']],
        }
        audited = fine_audit['branch_impact'][branch]
        for derived, independent in {
            'clips': 'total_selected_clips', 'supported_mapping_clips': 'supported_mapping_clips',
            'unresolved_clips': 'quarantined_clips', 'target_frames': 'supported_clip_target_frames',
            'clips_with_changed_indices': 'clips_with_changed_frame_indices',
            'changed_frame_indices': 'changed_selected_frame_indices',
        }.items():
            require(branches[branch][derived] == audited[independent], f'Branch impact differs: {branch}/{derived}')
        require(sorted(branches[branch]['out_of_bounds_ids']) == sorted(audited['mapped_clip_out_of_bounds_ids']),
                'Mapped bounds differ from independent audit')
        delta_counts = Counter()
        for item in selected:
            record, stem = record_lookup[item['id']]
            delta = refined[stem]['summary']['median_offset_seconds']
            fps, start = record['video_fps'], record['start_time']
            delta_counts.update(round((start+delta+k/25)*fps)-round((start+k/25)*fps)
                                for k in range(record['frames']))
        require(sum(delta_counts.values()) == branches[branch]['target_frames'] and
                sum(count for delta, count in delta_counts.items() if delta != 0) == branches[branch]['changed_frame_indices'],
                'Frame-index delta histogram differs from audited count')
        branches[branch]['source_frame_index_delta_histogram'] = {str(k): v for k, v in sorted(delta_counts.items())}
    storage = read(OUT/'training_storage_scope.json')
    preserved = {
        'data_cache/datasets/aihub_timed_v4/manifest.json': '214ce935150b367a15f24f0896901d34c54c6d4fe73d493a9d397c4c94fe59a6',
        'results/step17-expansion/run01/A_keep256/step_6400.pt': 'cd414f5976ade4ea535bf116549cc9b66372c27447879626d54e294ffad7ddba',
        'results/step17-expansion/run01/B_expand512/step_6400.pt': 'b82d9504191fab5c72877a5bc4987c8487107dfca2dc6d6cb24520429f5115cb',
        'experiments/legacy_recheck_step05/results/result.json': '2dc94c4f1209fbe00dacb54e671ab65f2434f76bc42cd01715a95cbb34877eba',
    }
    for name, expected in preserved.items():
        require(sha(ROOT/name) == expected, f'Preserved evidence changed: {name}')
    summary = {
        'schema': 'training-step06-final-summary-v1',
        'scope': 'Stage17 A256/B512 selected training recordings only; no retraining or input-cache replacement',
        'evidence': {name: binding(BASE/path) for name, path in {
            'inventory': 'training_inventory.json', 'original_plan': 'plan.json',
            'refinement_plan': 'refinement_plan.json', 'original_audit': 'results/independent_audit.json',
            'refinement_audit': 'results/independent_refined_audit.json',
            'exposure_audit': 'training_exposure_audit.json', 'storage': 'results/training_storage_scope.json',
        }.items()},
        'cohort': inventory['summary'],
        'original_8k': {'supported': original_progress['supported_sources'], 'quarantined': original_progress['quarantined_sources']},
        'refined_48k_grid': {
            'supported': len(supported), 'quarantined': len(quarantined), 'anchors': len(anchors),
            'offset_distribution': [{'seconds': k, 'sources': v} for k, v in sorted(offsets.items())],
            'minimum_correlation': min(a['score']['absolute_correlation'] for a in anchors),
            'minimum_far_peak_ratio': min(a['score']['peak_ratio'] for a in anchors),
            'maximum_within_source_offset_range_seconds': max(r['summary']['offset_range_seconds'] for r in supported),
            'sources_at_least_100ms': sum(abs(r['summary']['median_offset_seconds']) >= .1 for r in supported),
            'quarantined_source_stems': [r['source_stem'] for r in quarantined],
        },
        'branch_frame_index_impact': branches,
        'preserved_sha256': preserved,
        'limits': [
            'Interpolated48k search grid is not new48k signal sampling or a measurement-accuracy guarantee.',
            'Three audio anchors per recording do not verify every transcript, lip/audio synchrony or every internal edit.',
            'Sparse media decoding, metadata and excerpt hashes do not validate every byte of whole media files.',
            'Future mappings assume JSON sentence times refer to its named original WAV.',
            'No causal conclusion about model errors follows without a separately controlled learning experiment.',
        ],
    }
    save_once(OUT/'summary.json', summary)
    mapping = {
        'schema': 'training-step06-audited-audio-clock-candidates-v1',
        'status': 'candidates_only_not_applied_to_labels_caches_or_models',
        'summary': binding(OUT/'summary.json'),
        'time_formula': 'video_time_seconds = label_time_seconds + offset_seconds',
        'assumption': 'JSON sentence times refer to the named original WAV; audio correspondence alone does not verify transcript truth or lip/audio sync',
        'scope': 'Only supported recordings within the exact59 audited originals are included; never extrapolate to a speaker, package or missing source',
        'sources': [{
            'source_stem': r['source_stem'], 'offset_seconds': r['summary']['median_offset_seconds'],
            'original_label_sha256': sources[r['source_stem']]['expected_label_sha256'],
            'original_video_size_bytes': sources[r['source_stem']]['video']['size_bytes'],
            'refined_result': binding(OUT/f"refined_sources/{r['source_stem']}.json"),
            'training_utterance_ids': sources[r['source_stem']]['branch_sample_ids']['B_expand512'],
        } for r in supported],
        'unresolved_source_stems': [r['source_stem'] for r in quarantined],
        'originals_modified': False,
    }
    save_once(OUT/'audited_time_mapping_candidates.json', mapping)

    fine = summary['refined_48k_grid']
    offset_text = ', '.join(f"{k*1000:+.3f}ms가 {v}개" for k, v in sorted(offsets.items()))
    media_bytes = sum(inventory['summary']['source_file_bytes_if_fully_materialized'][k] for k in ('video','audio'))
    pcm_bytes = original_audit['verified']['registered_PCM_bytes']
    branch_rows = '\n'.join(
        f"| {branch} | {v['clips']} | {v['supported_mapping_clips']} | {v['clips_with_changed_indices']} | {v['changed_frame_indices']:,}/{v['target_frames']:,} | {len(v['out_of_bounds_ids'])} |"
        for branch, v in branches.items())
    data_rows = '\n'.join(
        f"| {branch} | {s['clips']} | 8 | 59 | {s['sampled_duration_minutes']:.2f}분 | {s['existing_cache_npz_bytes']/1e9:.3f}GB |"
        for branch, s in storage['branches'].items())
    frame_delta_text = ', '.join(f'{int(k):+d}프레임: {v:,}개' for k, v in branches['B_expand512']['source_frame_index_delta_histogram'].items())
    report = f'''# 새 계획 6단계 — 실제 학습59녹화 시간축 점검

기존 17단계 A(256문장)·B(512문장)가 실제 학습한 **8명·59개 녹화**의 원본 WAV와 MP4 내 음성을 비교했다. 원본59개 모두 MP4·WAV·JSON 보유를 확인했고, 녹화마다30·150·270초에서 검사해 총177쌍의 음성 구간을 확인했다. 전체 v4 train1,000개나 과거 모든 학습 이력으로 범위를 확대 해석하지 않는다.

음성은 데이터 시간축 검증에만 사용했다. 최종 프로그램의 입력에 음성을 추가한 것이 아니며, 최종목표는 계속 **웹캠 입술 영상만으로 한국어 텍스트를 추출**하는 것이다.

**정밀 계산 결과: {len(supported)}/59개에서 일정한 시간 이동을 지지했고, 보류 {len(quarantined)}개다.** 이동량 분포는 {offset_text}이다. 지지된 녹화 중 절댓값0.1초 이상은 {fine['sources_at_least_100ms']}개다. 이0.1초는 보고용 기준이며 무해함을 보장하는 기준이 아니다. 검증 자료 C025의 −8.362초를 이번 학습 원본에 적용하지 않았다.

이 결과만으로 기존 모델의 낮은 성능을 전부 시간축 탓으로 설명할 수는 없다. 작은 이동도 선택 프레임을 바꾸므로 보정 후보는 기록했지만, 이번 단계에서 캐시를 다시 만들거나 학습하지 않았다. [녹화59개 전체 결과](06_녹화별_점검결과.md)를 함께 제공한다.

## 검사 대상과 데이터 의미

| 학습 분기 | 문장 | 화자 | 녹화 | 25fps 입력 환산 길이 | 기존 압축 ROI 캐시 |
|---|---:|---:|---:|---:|---:|
{data_rows}

A의256문장은 B의512문장에 포함된다. 두 분기의 화자와 녹화 집합은 동일했다. 실제 성공한 optimizer update의 노출 기록도 전부 대조했다(A 문장당49~50회, B24~25회; 각 분기 총12,796회 노출). 따라서 이전의256→512 확대는 새로운 화자·새 녹화 환경 추가 실험은 아니었다. 적은 자료량과 제한된 다양성은 후속 검증할 요인이지만, 이번 시간 감사만으로 성능 정체의 단일 원인이라고 확정하지 않는다.

## 원래 검사와 후속 정밀 계산을 구분한 이유

원래 계획은 입력을8kHz로 추출해1/8000초(125μs) 간격으로 전 구간을 탐색했다. 상관계수0.99 이상, 멀리 떨어진 두 번째 후보 대비1.5배 이상, 충분한 신호 에너지, 검색 경계와0.5초 이상 거리, 세 구간의 이동량 범위0.02초 미만을 모두 요구했다. 이 판정은 **통과 {original_progress['supported_sources']}개·보류 {original_progress['quarantined_sources']}개**로 그대로 보존했다. 보류가 곧 원본 파일 손상이나 문장 연결 오류를 뜻하지 않는다.

먼저 확인한 두 녹화에서125μs 계산 격자의 영향이 관찰됐다. 이를 확인한 뒤 별도 후속 계획을 고정하여 **전체59개**에 같은 정밀 계산을 적용했다. 기존 필터를 거친 저장 PCM을6배 보간하고, 동일한36초 탐색창 전체를1/48000초 간격으로 검색했다. 결과가 나쁜 파일만 골라 계산하거나0.99 기준을 낮추지 않았다. 원래 계획에 포함됐던 절차로 소급해서 서술하지도 않는다.

최종177구간의 최소 상관계수는 **{fine['minimum_correlation']:.6f}**, 최소 먼 후보 대비 비율은 **{fine['minimum_far_peak_ratio']:.3f}배**였다. 지지된 녹화 내 세 지점의 최대 이동량 범위는 {fine['maximum_within_source_offset_range_seconds']:.3g}초였다(부동소수점 표현 오차 포함). 48kHz 보간 **검색 격자**이지48kHz로 새 신호를 관측했거나 그 수준의 측정 정확도를 입증한 것은 아니다.

## 시간 이동과 학습 프레임에 미치는 영향

후속 보정 후보의 부호는 `영상 시각 = 라벨 시각 + 이동량`이다. 이는 JSON의 문장 시각이 연결된 원본 WAV 기준이라는 가정에 따른다. 녹화별 측정 결과만 저장했으며 화자별·압축 묶음별로 일괄 적용하지 않는다.

| 분기 | 전체 문장 | 매핑 지지 문장 | 선택 번호가 바뀌는 문장 | 바뀌는 프레임 번호/전체 | 보정 시간 범위 초과 문장 |
|---|---:|---:|---:|---:|---:|
{branch_rows}

기존25fps 추출식 `round((start + k/25) * 실제FPS)`에서 start에 원본별 이동량을 더했을 때의 **프레임 번호 계산**이다. 실제 픽셀의 차이나 모델 정확도의 변화를 측정한 수치는 아니다. 보정 전후 캐시·성능 비교는 별도의 실험이 필요하다.

B의 원본 프레임 번호 이동 분포는 {frame_delta_text}이다. 번호가 바뀌는 프레임 비율이 크더라도 이것을 수초 단위로 잘못 연결된 문장 비율로 해석하지 않는다.

## 대용량 자료를 다룬 방식

관련 MP4+WAV 원본은 목록상 **{media_bytes/1e9:.2f}GB**지만 전체를 프로젝트에 복사하지 않았다. 필요한 음성 구간354개만 추출했고, 저장한 PCM은 헤더 포함 **{pcm_bytes/1e6:.2f}MB**다. 추가 정밀 계산은 이 로컬 PCM만 읽었다. 다만 Drive가 부분 읽기에도 원본 일부 또는 전체를 캐시할 수 있으므로, PCM 용량과 실제 다운로드/캐시 용량이 같다고 주장할 수 없다. 실행 전 여유 공간을 검사했고 매 원본 읽기 전10GiB 여유 공간 하한을 확인했다.

이미 있는512문장 ROI 캐시는 약{storage['branches']['B_expand512']['existing_cache_npz_bytes']/1e9:.3f}GB다. 원본을 보관하는 공간과 반복 학습용 작은 캐시를 나누어 다루는 방식이 유효하다. 이번 단계에는 새 학습 데이터나 AIHub 재다운로드가 필요하지 않았다. 이전18단계 TS2·TS20·TS200의 잘린 파일 복구 여부를 이번 결과로 판단하지 않는다.

## 검증과 한계

원본별 읽기 전후 크기·수정시각, JSON 전체 SHA-256 및 선택 문장의 ID·원문·시각, MP4 메타데이터·보수적인 재생 시간, WAV 헤더를 확인했다. 추출 음성의 해시·길이·형식·명령과 전체177구간의 탐색 결과를 별도 코드로 다시 계산했다. 원래8kHz 검사와 후속 정밀 검사 모두 독립 검산을 통과했다. 알려진 지연·DC·극성·무음·반복음·경계와 증거 변경 시 차단 등을 확인하는 신호·무결성18검사(원12+정밀6)도 통과했다.

음성끼리 맞는다는 사실은 모든 문장 전사의 정확성, 입술과 음성의 동기, 세 검사 지점 사이의 모든 편집 여부를 보장하지 않는다. 부분 디코딩과 해시 검사는 전체 원본 파일의 모든 바이트 검사도 아니다. 같은 개발20문장을 반복 사용했으므로 이전/이후 개발 성능은 최종 일반화 성능과 구분해야 한다. 최종test50의 영상·캐시를 읽거나 모델로 평가하지 않았다.

기존 v4 manifest, A/B 가중치와5단계 결과의 SHA-256이 그대로임을 확인했다. 신규 모델 추론·학습·서비스 교체는 없었다. 따라서5단계의 두 기존 모델 CER84.56%·0/20 결과는 바뀌지 않았다.

## 다음 진행

큰 시간 이동이 전반적인 학습 실패 원인이라는 가설과 실제 확인된 작은 원본별 이동을 구분한다. 검증된 이동량을 새 입력에 적용할 수 있도록 후보표를 준비했으며, 기존 자료는 재현용으로 유지한다. 다음 실험의 구체적인 범위는 [다음 단계 제안](06_다음단계_제안.md)에 기록한다. 새 원본이 필요해지는 시점에는 필요한 파일과 이유·최소 용량부터 알린다.

## 재현 파일

모든 실행 파일과 JSON은 `experiments/training_alignment_step06/`에 있다.

| 파일 | 역할 |
|---|---|
| `training_inventory.json`, `training_exposure_audit.json` | 실제 학습 원본·문장·노출 이력 |
| `plan.json`, `audit_sources.py` | 원래 고정 검사 조건과 실행 |
| `results/sources/`, `results/pcm/` |59개 원래 결과와354개 추출 음성 |
| `results/fractional_probe.json`, `refinement_plan.json` | 후속 검사의 사후 진단 계기와 별도 고정 계획 |
| `refine_alignment.py`, `results/refined_sources/` | 전체59개 정밀 결과 |
| `independent_audit.py`, `independent_refined_audit.py` | 독립 전체 탐색·해시·집계 검산 |
| `results/independent_audit.json`, `results/independent_refined_audit.json` | 검산 결과 |
| `results/summary.json`, `results/audited_time_mapping_candidates.json` | 집계와 적용 전 매핑 후보 |
| `write_reports.py` | 확정된 결과에서 이 문서와59개 표 재생성 |

원래 계획 SHA-256: `{sha(BASE/'plan.json')}`  
후속 계획 SHA-256: `{sha(BASE/'refinement_plan.json')}`  
집계 SHA-256: `{sha(OUT/'summary.json')}`  
매핑 후보 SHA-256: `{sha(OUT/'audited_time_mapping_candidates.json')}`
'''
    DOCS.mkdir(exist_ok=True)
    (DOCS/'06_실행결과.md').write_text(report, encoding='utf-8')
    rows = ['# 새 계획 6단계 — 녹화59개 전체 점검 결과', '',
            '원래8kHz 판정과 전체48kHz 보간 격자 후속 판정을 모두 보존한다. 시간은 MP4−원본WAV이며, 매핑 후보를 기존 라벨/캐시에 적용하지 않았다. [해석과 한계](06_실행결과.md).', '',
            '| 녹화 | A/B 문장 | 원래 판정 | 원래 최소 상관 | 후속 판정 | 후속 이동(ms) | 후속 최소 상관 | 세 지점 범위(ms) |',
            '|---|---:|---|---:|---|---:|---:|---:|']
    for stem in sorted(sources):
        old, new, src = originals[stem], refined[stem], sources[stem]
        sm, om = new.get('summary', {}), old.get('summary', {})
        label = lambda v: '지지' if v == 'supported' else '보류'
        rows.append(f"| {stem} | {len(src['branch_sample_ids']['A_keep256'])}/{len(src['branch_sample_ids']['B_expand512'])} | {label(old['status'])} | {om.get('minimum_correlation', float('nan')):.6f} | {label(new['status'])} | {sm.get('median_offset_seconds', float('nan'))*1000:+.6f} | {sm.get('minimum_correlation', float('nan')):.6f} | {sm.get('offset_range_seconds', float('nan'))*1000:.3g} |")
    rows += ['', '## 보류 내역', '']
    rows += [f"- {r['source_stem']}: {r.get('reason', '사유 미기재')}" for r in quarantined] or ['후속 정밀 판정에서 보류된 녹화 없음. 원래 보류 판정과 원인은 각 원결과 JSON에 보존했다.']
    (DOCS/'06_녹화별_점검결과.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    pending_docs = []
    for name, prefix in [('README.md', 'docs/'), ('CLAUDE.md', 'docs/'),
                         ('docs/기획서_완성본.md', ''), ('docs/설계문서_완성본.md', '')]:
        path = ROOT/name
        original_bytes = path.read_bytes()
        lines = original_bytes.decode('utf-8').splitlines(keepends=True)
        positions = [i for i, line in enumerate(lines) if line.startswith('> **2026-09-24')]
        require(positions, f'Current status block not found: {name}')
        index = positions[0]
        ending = '\r\n' if lines[index].endswith('\r\n') else '\n'
        lines[index] = (
            f'> **2026-09-24 현재 상태: 새 계획 6단계 완료·독립 검산 통과.** '
            f'실제17단계 A256/B512의 학습8명·59녹화에서 원본 WAV/MP4 시간축177구간을 확인했다. '
            f'원래8kHz 판정을 보존한 별도 정밀 검사에서 {len(supported)}/59녹화의 매핑을 지지했으며 이동량은 {offset_text}이다. '
            f'지지된 녹화 중 절댓값0.1초 이상은 {fine["sources_at_least_100ms"]}개다. 원본별 매핑 후보를 별도 저장했고 기존 라벨·v4캐시·가중치는 유지했다. '
            '음성 시간 매핑은 전체 전사·입술 동기·모델 성능을 보장하지 않는다. 두 학습군은 같은8명·59녹화이며 B의 영상 입력은 약38.95분이다. '
            '다음 제안은 보정 입력을 공통 사용한 기존 compact의 기본 학습/시각 특징 전달 대조다. '
            '새5단계의 기존 A/B 문자 CER84.56%·0/20 결과는 그대로이며 추가 학습·test50·서비스 교체·18단계 손상 복구는 이번 범위에 없다. '
            f'[6단계 결과]({prefix}공식모델_도입검증/06_실행결과.md) · '
            f'[다음 단계 제안]({prefix}공식모델_도입검증/06_다음단계_제안.md).' + ending)
        pending_docs.append((path, original_bytes, ''.join(lines).encode('utf-8')))
    for path, original_bytes, updated in pending_docs:
        require(path.read_bytes() == original_bytes, f'Doc changed during report generation: {path}')
        path.write_bytes(updated)
    print(json.dumps({'summary': summary['refined_48k_grid'], 'branches': branches,
                      'report': str(DOCS/'06_실행결과.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
