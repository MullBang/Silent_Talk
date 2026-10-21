"""Produce compact Korean findings and explicit time-review candidates."""
from collections import Counter
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
rows = json.loads((base / "label_records.json").read_text(encoding="utf-8"))
anomalies = json.loads((base / "time_anomaly_details.json").read_text(encoding="utf-8"))
selection = json.loads((base / "selection_impact.json").read_text(encoding="utf-8"))
exclusions = []
for row in anomalies["records"]:
    for issue in row["time_issues"]:
        exclusions.append({"id": row["stem"] + f"_{issue['sentence_id']:03d}",
            "source_stem": row["stem"], "speaker": row["identity"]["speaker"],
            "angle": row["identity"]["angle"], "recording": row["identity"]["recording"],
            "label_archive": row["archive"], "label_member": row["member"],
            "label_sha256": row["label_sha256"], **issue,
            "status": "exclude_from_unmapped_preparation_pending_timing_review"})
frontal = [r for r in exclusions if r["angle"] == "A"]
document = {"schema": "lipnet-label-time-review-candidates-v1", "audit_date": "2026-10-02",
    "scope": "Original labels whose uncorrected end time exceeds the encoded video track duration",
    "original_files_modified": False,
    "policy": "Candidate review/exclusion list only. Do not silently clip timestamps, change transcripts, or copy C025's previously measured offset to other recordings. Existing independently audited mappings remain separate.",
    "not_exhaustive_semantic_error_list": True,
    "annotation_count_all_angles": len(exclusions), "annotation_count_frontal": len(frontal),
    "unique_recording_sentence_pairs": len({(r["recording"], r["sentence_id"]) for r in exclusions}),
    "max_end_past_video_seconds": max(r["end_past_video_seconds"] for r in exclusions),
    "max_frontal_end_past_video_seconds": max(r["end_past_video_seconds"] for r in frontal),
    "boundary_overshoots_at_most_one_frame_count": sum(r["end_past_video_seconds"] <= 1001 / 30000 for r in exclusions),
    "by_speaker": {speaker: {"annotations": sum(r["speaker"] == speaker for r in exclusions),
        "min_end_past_video_seconds": min(r["end_past_video_seconds"] for r in exclusions if r["speaker"] == speaker),
        "max_end_past_video_seconds": max(r["end_past_video_seconds"] for r in exclusions if r["speaker"] == speaker)}
        for speaker in sorted({r["speaker"] for r in exclusions})},
    "records": exclusions, "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(base / "sentence_time_exclusions.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
table = []
for package in ("TL1", "TL10", "TL2", "TL20", "TL200"):
    rr = [r for r in rows if Path(r["archive"]).stem == package]
    table.append(f"| {package} | {len(rr)} | {sum(bool(r['errors']) for r in rr)} | {sum(r['detail']['sentence_count'] for r in rr if r['detail']['identity']['angle'] == 'A'):,} |")
report = """# 학습 자료 라벨·영상 메타데이터 무결성 검사

검사 기준일: 2026-10-02. 원본 TAR·JSON·영상 및 기존 학습/평가 명세는 수정하지 않았다.

5개 TL TAR의 JSON 1,500개를 전부 읽고 UTF-8/JSON 파싱, 파일명·화자·각도 연결, 문장 ID·전사 형식·시간, 입술/얼굴 좌표를 검사했다. 5개 TS TAR의 MP4 1,500개에서는 moov 메타데이터를 읽어 해상도·실제 FPS·영상 트랙 길이·encoded sample 수와 라벨을 대조했다. 영상 전체 디코딩은 별도 검사다.

## 결과

모든 JSON은 파싱되었고 참조 영상 누락·중복, 화자/각도 연결 오류, 영상 해상도 불일치가 없었다. 입술 및 얼굴 bbox는 각각 13,730,079개를 검사했으며 모든 좌표가 유효하고 모든 영상의 encoded sample 수와 개수가 일치했다. 서로 다른 영상은 1,500개, 녹화는 300개, 화자는 25명이다. 녹화별 5개 각도에 동일한 문장 ID·전사·시간이 기재되어 있었다. 전체 각도 중복을 포함한 문장 주석은 63,140개, 정면 A 기준은 12,628개다.

| 라벨 묶음 | JSON | 시간 오류가 있는 JSON | 정면 문장 주석 |
|---|---:|---:|---:|
""" + "\n".join(table) + """

시간 범위 오류는 JSON 68개에 포함된 88개 문장 주석이다. 정면만 보면 C025의 12개 JSON에 포함된 16문장이다. 이는 파일 전체가 재생 불가능하거나 해당 파일의 모든 전사가 틀렸다는 뜻은 아니다. 다만 C025는 이전에 별도 WAV/MP4 파형 검사로 시간축 차이가 확인된 화자이므로, 마지막 문장만 빼고 나머지를 자동으로 정상 처리해서도 안 된다.

- C025: 12녹화 × 5각도, JSON 60개/문장 주석 80개. 종료 시각이 영상 끝을 0.600179~7.550433초 초과한다. 정면 A006은 실제 영상 301.868233초인데 ID42는 305.450667초, ID43은 309.418667초에 끝난다고 되어 있다. A007 실제 길이는 302.568933초다. 과거 −8.362초 매핑은 독립적으로 확인한 A006/A007에 관한 기록이며 이번에 다른 녹화로 확대하지 않았다.
- C002: C/E/G/I 각도 005 녹화의 ID50이 각각 2.001933초 초과한다. 정면 A는 이 오류에 포함되지 않는다.
- C032: B/D/F/H 각도 011 녹화의 ID38이 3.293133~3.426600초 초과한다. 정면 A는 이 오류에 포함되지 않는다.

한 프레임 이하의 단순 끝 경계 초과는 0건이다. 위 88개 후보는 `sentence_time_exclusions.json`에 ID·원본 라벨 해시·시각·초과량을 저장했다. 독립적인 시간 매핑 확인 전에는 해당 원시 시각으로 준비하지 않으며, 자동 잘라내기나 전사 교체는 수행하지 않았다. 이 목록은 모든 의미/동기 오류를 찾아냈다는 목록이 아니다.

## 현재 선택된 학습 문장에 미치는 영향

현재 기존 stage17 B512의 512문장/59개 원본과 신규 stage18 후보 120문장/30개 원본은 선택된 문장에 위 메타데이터 오류가 없었다. 59+30개 원본 라벨의 SHA-256은 기존 고정 명세와 모두 일치했다. 기존 선택 명세를 고치거나 새 학습을 실행하지 않았다. 결과는 `selection_impact.json`이다.

이는 라벨 파일의 바이트와 구조·영상 범위가 맞다는 결과다. 전사가 실제 발화와 일치하는지, 입술과 소리가 동기화되었는지, 신규 데이터의 WAV 시각을 MP4 시각으로 바로 쓸 수 있는지까지 검증한 결과가 아니다. 실제 사용 전에는 영상 전체 디코딩 결과와 녹화별 시간 매핑 검증을 함께 적용해야 한다.

## 오류와 구분한 경고

- 모든 JSON의 FPS=30은 영상의 29.97002997fps와 다르다. 기존 전처리처럼 실제 영상 시간축을 사용해야 한다.
- 비정면 JSON 1,200개는 자기 각도 이름의 WAV를 기재하지만 실제 WAV는 녹화별 A 한 개씩, 총 300개다. A WAV가 공유 음성 후보라고 기록했으며 각 카메라와 시간 정렬이 같다고 가정하지 않았다.
- 같은 녹화에서 반복 전사가 있는 JSON은 130개다. 반복 발화일 수 있어 손상으로 판정하지 않았다.
- JSON의 정수 단위 명목 영상 길이와 실제 길이가 1.1초 넘게 다른 것은 148개다. 이 중 정면은 C025 12개뿐이다. 비정면은 별도 카메라의 시작/종료 차이 등 원인 확인이 필요하다.

## 재현 및 검증

`audit_labels.py`의 합성 검사 통과, 확장된 TS2 영상 3개의 독립 OpenCV 메타데이터와 자체 moov 파서 대조 통과. `metadata_crosscheck.json`에 결과를 저장했다. 오류 라벨 68개를 다시 읽어 초기 검사와 SHA-256이 같은 것을 확인한 후 실제 오류 시각을 `time_anomaly_details.json`에 기록했다. 원본 TAR는 읽기 전후 크기/수정시각이 같았다. 실행 중 추가한 선택 영향 함수는 새 Python 프로세스에서 별도로 실행했으며 생산 코드 범위는 `execution_provenance.json`에 명시했다.

공식 제공자 체크섬과의 대조, 전체 영상 디코딩, 전사 내용 수동 검수는 이 라벨 검사 자체의 범위에 포함되지 않는다.
"""
(base / "summary_ko.md").write_text(report, encoding="utf-8")
print(json.dumps({k: document[k] for k in ("annotation_count_all_angles", "annotation_count_frontal", "unique_recording_sentence_pairs", "max_end_past_video_seconds", "boundary_overshoots_at_most_one_frame_count", "by_speaker")}, ensure_ascii=False))
