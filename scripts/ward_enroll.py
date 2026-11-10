"""병실 명령 등록(내 입 모양 템플릿) 오프라인 평가 (기획 9.1 등록 ①, 10.2 등록 효과).

세션 임시 실험(command_idea/exp/roi_extract.py, enroll_exp/features.py·analyze.py, p02 등록 분석)을 저장소로 옮긴 것이다.
  roi      : .venv-olkavs-roi  촬영 폴더 → 서비스 녹화 경로와 같은 설정의 입술 창 NPZ
             (extract_segments: 창 12초, 최소 창 1.0초, 자르기 탐색 6초, resample 없음, utterance=True)
  features : .venv-olkavs      창이 하나인 클립 → CTC 직전 은닉 특징(T x 256) NPZ + 병실 명령 20개 후보별 토큰당 점수
             (서비스 recognize와 같은 계산이라 1등·점수·차이가 서비스 phrase와 같아야 한다 → ward_eval_tools가 대조)
  apply    : numpy             사람별 등록 템플릿(기본: 소리 내어 1회 = 등록용 촬영)과 나머지 클립의 DTW 거리, D를 주면 판정
  run      : roi → features를 각 가상환경으로 실행

봉인 자료(P02, ward_eval)는 모든 단계에서 --allow-sealed --freeze <커밋된 status frozen 동결 파일>이 있어야 한다.
동결 파일은 클립 기록·특징을 읽기 전에 확인하고, 그 SHA-256·커밋과 실행 때 git HEAD를 각 출력 파일 머리에 남긴다.
apply --freeze는 동결 파일의 기준값·긴급 가능성 하한·D·템플릿 방식을 쓰며, 다른 --template을 주면 거부한다.
이 도구는 오프라인 등록(보조 결과)이다. 앱이 실제로 쓰는 서버 등록은 ward_service_batch.py --enroll-profile로 채점한다.
특징(서버 hidden_out과 같은 계산)과 DTW(backend/services/ward_templates.py와 같은 코드)는 서버 등록과 같아야 한다.
영상·특징·결과는 저장소 밖 또는 experiments/ 아래(gitignore)에 둔다. 등록 특징은 P번호로만 저장하고 시연 뒤 지운다(기획 11).

사용 예:
  py -3.11 scripts/ward_enroll.py run --clips "C:/.../웹캠 영상 P01/ward_tune" --out experiments/ward_enroll_20261111/P01_ward_tune
  py -3.11 scripts/ward_enroll.py run --clips "C:/.../웹캠 영상 P01" --out experiments/ward_enroll_20261111/P01_sentences
  py -3.11 scripts/ward_enroll.py apply --enroll-dir experiments/ward_enroll_20261111/P01_ward_tune \
      --negatives experiments/ward_enroll_20261111/P01_sentences --out experiments/ward_enroll_20261111/P01_enroll.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import olkavs_transcribe as ot  # noqa: E402
import pilot_evaluate as pe  # noqa: E402
from ward_tools import common  # noqa: E402

ROI_SCHEMA = "ward-enroll-roi-v1"
FEATURES_SCHEMA = "ward-enroll-features-v1"
APPLY_SCHEMA = "ward-enroll-v1"
FEATURE_NAME = common.FEATURE_NAME
DISTANCE_NAME = common.DISTANCE_NAME
TEMPLATES = common.ENROLL_TEMPLATES
DEFAULT_TEMPLATE = "voiced"


def run_head(gate: dict[str, Any] | None) -> dict[str, Any]:
    """출력 파일 머리에 남길 실행 정보 (만든 시각, git HEAD, 동결 파일)."""
    return {"created": datetime.now(timezone.utc).isoformat(), "git_head": common.git("rev-parse", "HEAD"),
            "freeze": ({k: gate[k] for k in ("path", "sha256", "status", "commit", "commit_time", "sealed_ok")}
                       if gate else None)}


def same_freeze(index: dict[str, Any], gate: dict[str, Any] | None, what: str) -> None:
    """봉인 자료의 앞 단계 출력이 같은 동결 파일로 만들어졌는지 확인한다.

    Raises:
        SealedDataError: 앞 단계 출력에 동결 기록이 없거나 SHA-256이 다를 때
    """
    prev = (index.get("freeze") or {}).get("sha256") if isinstance(index.get("freeze"), dict) else None
    if gate is None or prev != gate.get("sha256"):
        raise common.SealedDataError(f"{what}: 봉인 자료의 앞 단계 출력이 이 동결 파일로 만들어지지 않았습니다(freeze.sha256 {prev!r})")


def clip_info(row: dict[str, Any]) -> dict[str, Any]:
    """클립 기록에서 결과 파일에 남길 정보."""
    return {k: row.get(k) for k in ("file", "participant", "session", "kind", "condition", "sentence_id", "rep", "posture",
                                    "duration_ms")}


def is_template(row: dict[str, Any], template: str) -> bool:
    """등록 템플릿 클립인가: 병실 묶음의 서 있는 자세 명령 중 소리 내어 1회(voiced), 소리 없이 1회(silent1), 또는 둘 다(both).

    규칙은 common.is_template에 있다(서버 등록을 채우는 ward_service_batch.py --enroll-from voiced와 같은 클립을 고른다).
    """
    return common.is_template(row, template)


def pseudo_row(rec: dict[str, Any]) -> dict[str, Any]:
    """특징 기록 → metrics.prepare가 읽는 서비스 결과 행 모양 (오프라인 글자 순위를 phrase로 쓴다)."""
    rank = rec.get("text_rank")
    if rec.get("status") == "failed":
        return {**clip_info(rec), "status": "error", "results": None}  # 입술 추출·특징 계산 실패
    if rec.get("status") == "ok":
        # 후보는 병실 명령 목록(commands.json)이므로 서비스 결과와 같게 set을 붙인다
        phrase = {**rank, "set": "ward"} if isinstance(rank, dict) else None
        window = {"start_ms": rec.get("start_ms"), "end_ms": rec.get("end_ms"), "text": "", "phrase": phrase}
        results = [window]
    elif rec.get("status") == "no_speech":
        results = [{"text": common.NO_SPEECH_TEXT}]
    elif rec.get("status") == "lip_fail":
        results = [{"text": common.FAIL_TEXT}]
    else:  # no_result: 여러 창·일부 검출 실패 → 지정 문장 결과 없음
        results = [{"text": common.FAIL_TEXT}, {"text": ""}]
    return {**clip_info(rec), "status": "done", "results": results}


# ---------------------------------------------------------------------------
# roi (.venv-olkavs-roi)
# ---------------------------------------------------------------------------

def cmd_roi(args: argparse.Namespace) -> int:
    """촬영 폴더의 클립마다 서비스와 같은 설정으로 입술 창을 만든다 (이어 하기 지원)."""
    clips_dir, out = Path(args.clips).resolve(), Path(args.out).resolve()
    index_path = out / "roi_index.json"
    try:
        gate = common.freeze_gate(args.freeze, args.allow_sealed)
        common.require_unsealed([clips_dir, out], [], args.allow_sealed, "등록 평가 입술 추출")
        clips = pe.load_clips(clips_dir)
        common.require_unsealed([clips_dir, out], clips, args.allow_sealed, "등록 평가 입술 추출")
        pe.check_rows(clips)
        sealed = bool(common.sealed_reasons([clips_dir, out], clips))
        previous = {}
        if index_path.exists():
            prev_index = common.read_json(index_path)
            if sealed:
                same_freeze(prev_index, gate, "이어 하기")
            previous = {r["file"]: r for r in prev_index.get("rows", []) if r.get("status") == "ok"}
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    head = {"schema": ROI_SCHEMA, **run_head(gate), "clips_dir": str(clips_dir), "roi_settings": common.SERVICE_ROI}
    rows = []
    for clip in clips:
        if clip.get("status") != "ok":
            rows.append({**clip_info(clip), "status": clip.get("status")})
            continue
        stem = Path(clip["file"]).stem
        prev = previous.get(clip["file"])
        if prev and all(not s.get("npz") or Path(s["npz"]).exists() for s in prev.get("segments", [])):
            rows.append(prev)
            continue
        row: dict[str, Any] = clip_info(clip)
        t0 = time.perf_counter()
        try:
            res = ot.extract_segments(Path(clip["path"]), out / "roi" / stem, progress=None, duration_hint_seconds=None,
                                      **common.SERVICE_ROI)
            segs = res["segments"]
            row.update(status="ok", single=len(segs) == 1 and segs[0].get("status") == "ok" and bool(segs[0].get("npz")),
                       segments=[{k: s.get(k) for k in ("index", "start_ms", "end_ms", "frames", "status", "reason", "npz")}
                                 for s in segs],
                       speech={k: res["speech"].get(k) for k in ("enabled", "detected", "start_ms", "end_ms", "active_seconds",
                                                                  "trimmed_head_ms", "trimmed_tail_ms")})
        except Exception as exc:  # noqa: BLE001 - 클립 하나의 실패는 기록하고 계속한다
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(row)
        common.atomic_write_json(index_path, {**head, "rows": rows})
        print(json.dumps({"file": row["file"], "status": row["status"], "single": row.get("single"),
                          "segs": [(s["status"], s["start_ms"], s["end_ms"], s["frames"]) for s in row.get("segments", [])],
                          "s": row["seconds"]}, ensure_ascii=False), flush=True)
    common.atomic_write_json(index_path, {**head, "clips_listing": common.folder_fingerprint(clips_dir), "rows": rows})
    return 0


# ---------------------------------------------------------------------------
# features (.venv-olkavs)
# ---------------------------------------------------------------------------

def segment_status(row: dict[str, Any]) -> str:
    """입술 추출 결과 → ok(창 하나) / no_speech / lip_fail / no_result(여러 창·일부 실패) / failed."""
    if row.get("status") != "ok":
        return "failed"
    segs = row.get("segments") or []
    if row.get("single"):
        return "ok"
    if segs and all(s.get("status") == "no_speech" for s in segs):
        return "no_speech"
    if segs and all(s.get("status") == "failed" for s in segs):
        return "lip_fail"
    return "no_result"


def cmd_features(args: argparse.Namespace) -> int:
    """창이 하나인 클립의 은닉 특징과 명령 후보별 점수를 계산한다 (서비스 recognize와 같은 계산)."""
    out = Path(args.out).resolve()
    try:
        gate = common.freeze_gate(args.freeze, args.allow_sealed)
        common.require_unsealed([out], [], args.allow_sealed, "등록 평가 특징 계산")
        index = common.read_json(out / "roi_index.json")
        common.require_unsealed([out, index.get("clips_dir")], index.get("rows", []), args.allow_sealed, "등록 평가 특징 계산")
        if common.sealed_reasons([out, index.get("clips_dir")], index.get("rows", [])):
            same_freeze(index, gate, "입술 추출 결과")
        commands = common.load_commands()
    except (common.SealedDataError, ValueError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    import numpy as np
    import torch
    device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    bundle = ot.load_olkavs(device)
    model = bundle["model"]
    phrases = [{"id": cid, "text": commands["texts"][cid]} for cid in commands["ids"]]
    tokens = {p["id"]: ot.phrase_tokens(p["text"], bundle) for p in phrases}
    (out / "features").mkdir(parents=True, exist_ok=True)
    recs = []
    for row in index["rows"]:
        rec: dict[str, Any] = {**clip_info(row), "status": segment_status(row)}
        if rec["status"] == "ok":
            seg = row["segments"][0]
            try:
                frames = ot.read_frames(Path(seg["npz"]))
                video, _norm = ot.normalize_frames(frames)
                video = video.to(device)
                with torch.inference_mode():
                    feats = model.medium(model.encoder(video, torch.tensor([video.shape[2]], device=device), None, None))
                    if not torch.isfinite(feats).all():
                        raise ValueError("encoder 출력에 비정상 값")
                    logits = model.decoder.ctcdecoder(inputs=feats)
                    log_probs = torch.log_softmax(logits[0].double(), dim=-1).cpu()
                    hidden = feats[0].float().cpu().numpy()
                scores = [ot.ctc_forced_score(log_probs, tokens[p["id"]], blank=3) for p in phrases]
                npz = out / "features" / f"{Path(row['file']).stem}.npz"
                np.savez(npz, hidden=hidden)
                rec.update(frames=int(frames.shape[0]), start_ms=seg["start_ms"], end_ms=seg["end_ms"],
                           hidden_dim=int(hidden.shape[1]), features_npz=str(npz),
                           text_scores={p["id"]: (s["per_token"] if s["status"] == "ok" else None)
                                        for p, s in zip(phrases, scores)},
                           text_rank=ot.rank_phrases(phrases, scores))
            except (ValueError, RuntimeError, OSError) as exc:
                rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        recs.append(rec)
        print(json.dumps({"file": rec["file"], "status": rec["status"], "top": (rec.get("text_rank") or {}).get("id")},
                         ensure_ascii=False), flush=True)
    common.atomic_write_json(out / "features_index.json", {
        "schema": FEATURES_SCHEMA, **run_head(gate), "clips_dir": index.get("clips_dir"),
        "clips_listing": index.get("clips_listing"), "device": device, "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "checkpoint_sha256": ot.CHECKPOINT_SHA256, "commands": {k: commands[k] for k in ("version", "sha256", "ids")},
        "feature": FEATURE_NAME, "clips": recs})
    return 0


# ---------------------------------------------------------------------------
# apply (numpy)
# ---------------------------------------------------------------------------

def load_features(dirs: list[str], allow_sealed: bool, gate: dict[str, Any] | None = None,
                  listings: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """features_index.json 여러 개를 읽어 클립 기록을 합친다 (봉인 확인 포함).

    봉인 자료의 특징 폴더는 같은 동결 파일(gate)로 만들어졌어야 한다. listings를 주면 폴더마다 촬영 폴더 목록 지문을 더한다.
    """
    recs: list[dict[str, Any]] = []
    for d in dirs:
        common.require_unsealed([d], [], allow_sealed, "등록 평가")
        idx = common.read_json(Path(d) / "features_index.json")
        common.require_unsealed([d, idx.get("clips_dir")], idx.get("clips", []), allow_sealed, "등록 평가")
        if common.sealed_reasons([d, idx.get("clips_dir")], idx.get("clips", [])):
            same_freeze(idx, gate, f"특징 폴더 {d}")
        if listings is not None:
            listings.append({"dir": str(Path(d).resolve()), "clips_dir": idx.get("clips_dir"),
                             "listing_sha256": (idx.get("clips_listing") or {}).get("listing_sha256")})
        for r in idx.get("clips", []):
            recs.append({**r, "source_dir": str(Path(d).resolve())})
    return recs


def enroll_distances(recs: list[dict[str, Any]], template: str, command_ids: list[str],
                     load: Any = None) -> list[dict[str, Any]]:
    """사람별 템플릿을 모아 나머지 클립의 명령별 DTW 거리를 붙인다.

    Args:
        recs: features 기록 (ward 묶음 + 목록 밖 자료)
        template: is_template 방식
        command_ids: 명령 ID 순서
        load: 테스트용 주입 (기록 → (T, D) 배열). 기본은 features_npz의 hidden.
    Returns:
        기록 복사본에 is_template, tmpl_dists(템플릿이거나 창이 하나가 아니면 None), template_files를 더한 목록
    """
    import numpy as np
    from ward_tools import enroll

    def default_load(rec: dict[str, Any]) -> Any:
        with np.load(rec["features_npz"]) as data:
            return np.asarray(data["hidden"])

    load = load or default_load
    out = []
    for pid in sorted({str(r.get("participant")) for r in recs}):
        mine = [r for r in recs if str(r.get("participant")) == pid]
        temps: dict[str, list[Any]] = {cid: [] for cid in command_ids}
        tfiles: dict[str, list[str]] = {cid: [] for cid in command_ids}
        for r in mine:
            if r.get("status") == "ok" and is_template(r, template) and r.get("sentence_id") in temps:
                temps[r["sentence_id"]].append(enroll.unit_rows(load(r)))
                tfiles[r["sentence_id"]].append(r["file"])
        has_templates = any(temps.values())
        for r in mine:
            rec = dict(r)
            rec["is_template"] = is_template(r, template)
            rec["template_files"] = tfiles
            if r.get("status") == "ok" and not rec["is_template"] and has_templates:
                rec["tmpl_dists"] = enroll.template_distances(enroll.unit_rows(load(r)), temps)
            else:
                rec["tmpl_dists"] = None
            out.append(rec)
    return out


def cmd_apply(args: argparse.Namespace) -> int:
    """템플릿 거리 계산과 (D가 있으면) 판정. 결과는 ward_eval_tools calibrate·report가 읽는다."""
    from ward_tools import calibrate as cal
    from ward_tools import metrics as mt
    try:
        gate = common.freeze_gate(args.freeze, args.allow_sealed)
        common.require_unsealed([args.out], [], args.allow_sealed, "등록 평가")
        listings: list[dict[str, Any]] = []
        recs = load_features(args.enroll_dir + (args.negatives or []), args.allow_sealed, gate, listings)
        commands = common.load_commands()
        freeze = common.read_json(args.freeze) if args.freeze else None
        template = args.template or DEFAULT_TEMPLATE
        if freeze is not None:
            if args.D is not None and args.allow_sealed:
                raise ValueError("봉인 자료에는 동결 파일의 D만 씁니다(--D를 빼세요)")
            if (freeze.get("commands") or {}).get("sha256") != commands["sha256"]:
                raise ValueError("명령 목록이 동결 때와 다릅니다(commands.json SHA-256 불일치)")
            enr = freeze.get("enrollment") or {}
            if enr.get("max_distance") is not None:
                frozen_template = enr.get("template")
                if args.template and args.template != frozen_template:
                    raise ValueError(f"--template {args.template}이(가) 동결 파일의 템플릿 방식 {frozen_template}과 다릅니다")
                template = frozen_template
                if enr.get("feature") != FEATURE_NAME or enr.get("distance") != DISTANCE_NAME:
                    raise ValueError("동결 파일의 특징·거리 이름이 이 도구와 다릅니다")
            # 동결 파일에 D가 없으면 거리만 계산한다(판정 없음)
            rule = cal.rule_from_freeze(freeze, "full", with_enrollment=args.D is None and enr.get("max_distance") is not None)
            if args.D is not None:
                rule = replace(rule, max_distance=float(args.D))
        else:
            rule = mt.Rule(args.margin, args.score, frozenset(commands["emergency"]),
                           max_distance=float(args.D) if args.D is not None else None,
                           require_agreement=not args.no_agreement)
        if template not in TEMPLATES:
            raise ValueError(f"알 수 없는 템플릿 방식: {template}")
    except (common.SealedDataError, ValueError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    withd = enroll_distances(recs, template, commands["ids"])
    clips = []
    for rec in withd:
        out = {k: rec.get(k) for k in ("file", "participant", "session", "kind", "condition", "sentence_id", "rep", "posture",
                                       "status", "start_ms", "end_ms", "frames", "text_scores", "text_rank", "tmpl_dists",
                                       "is_template", "source_dir")}
        if rule.max_distance is not None:
            base = mt.prepare(pseudo_row(rec), commands, {"text_scores": rec.get("text_scores"),
                                                          "tmpl_dists": rec.get("tmpl_dists"),
                                                          "is_template": rec.get("is_template")})
            e = mt.evaluate(base, rule)
            out["decision"] = {k: e.get(k) for k in ("outcome", "chosen_id", "via", "text_pass", "conflict", "enroll_alarm",
                                                     "tmpl_top", "tmpl_dist")}
        clips.append(out)
    result: dict[str, Any] = {"schema": APPLY_SCHEMA, **run_head(gate),
                              "template": template, "feature": FEATURE_NAME, "distance": DISTANCE_NAME,
                              "rule": rule.describe(), "commands": {k: commands[k] for k in ("version", "sha256")},
                              "inputs": sorted({r["source_dir"] for r in recs}), "inputs_listing": listings,
                              "elapsed_seconds": None, "clips": clips}
    if rule.max_distance is not None:
        tests = [c for c in clips if c.get("tmpl_dists") is not None]
        bases = [mt.prepare(pseudo_row(c), commands, {"text_scores": c.get("text_scores"), "tmpl_dists": c.get("tmpl_dists")})
                 for c in tests]
        evals = mt.evaluate_all(bases, rule)
        result["summary_tests_only"] = mt.summarize_groups(evals, rule, commands)
        text_rule = replace(rule, max_distance=None)
        result["summary_tests_only_text_rule"] = mt.summarize_groups(mt.evaluate_all(bases, text_rule), text_rule, commands)
    result["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    common.atomic_write_json(args.out, result)
    n_t = sum(1 for c in clips if c.get("tmpl_dists") is not None)
    print(f"끝: 클립 {len(clips)}개 중 거리 계산 {n_t}개, 템플릿 {sum(1 for c in clips if c.get('is_template'))}개 → {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """roi → features를 각 가상환경으로 실행한다."""
    try:
        common.freeze_gate(args.freeze, args.allow_sealed)  # 두 단계도 각각 다시 확인한다
    except common.SealedDataError as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    script = str(Path(__file__).resolve())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    extra = (["--allow-sealed"] if args.allow_sealed else []) + (["--freeze", str(Path(args.freeze).resolve())]
                                                                  if args.freeze else [])
    rc = subprocess.run([str(ot.ROI_PYTHON), "-B", script, "roi", "--clips", args.clips, "--out", args.out, *extra],
                        env=env).returncode
    if rc != 0:
        return rc
    return subprocess.run([str(ot.INFER_PYTHON), "-B", script, "features", "--out", args.out, "--device", args.device, *extra],
                          env=env).returncode


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sealed_help = "봉인 자료(P02·ward_eval)도 처리 (--freeze에 커밋된 status frozen 동결 파일이 있어야 함)"
    freeze_help = "동결 파일 (봉인 자료에는 필수)"
    p = sub.add_parser("roi")
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-sealed", action="store_true", help=sealed_help)
    p.add_argument("--freeze", help=freeze_help)
    p.set_defaults(func=cmd_roi)
    p = sub.add_parser("features")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-sealed", action="store_true", help=sealed_help)
    p.add_argument("--freeze", help=freeze_help)
    p.set_defaults(func=cmd_features)
    p = sub.add_parser("run")
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-sealed", action="store_true", help=sealed_help)
    p.add_argument("--freeze", help=freeze_help)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("apply")
    p.add_argument("--enroll-dir", action="append", required=True, help="병실 묶음 features 폴더 (템플릿·시험, 여러 번 가능)")
    p.add_argument("--negatives", action="append", help="목록 밖 자료 features 폴더 (시험만, 여러 번 가능)")
    p.add_argument("--out", required=True)
    p.add_argument("--template", choices=TEMPLATES, default=None,
                   help=f"템플릿 방식 (기본 {DEFAULT_TEMPLATE}, --freeze에 D가 있으면 동결 값을 쓰고 다른 값은 거부)")
    p.add_argument("--freeze", help="동결 파일 (기준값·D·템플릿 방식을 여기서 읽음, 봉인 자료에는 필수)")
    p.add_argument("--D", type=float, default=None, help="등록 거리 D (없으면 거리만 계산, --freeze가 있으면 그 D)")
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--score", type=float, default=-3.40)
    p.add_argument("--no-agreement", action="store_true", help="글자 1등과 등록 1등이 달라도 고름(기획 9.3과 다름)")
    p.add_argument("--allow-sealed", action="store_true", help=sealed_help)
    p.set_defaults(func=cmd_apply)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
