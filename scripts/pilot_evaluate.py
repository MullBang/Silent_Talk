"""팀원 웹캠 파일럿 클립을 OLKAVS로 일괄 평가한다 (3단계).

입력 폴더에는 CapturePage가 저장한 `{가명}_{조건}_{문장ID}_{회차}.webm|mp4` 와 같은 이름의 `.json`(정답
문장·조건·프레임 로그)이 있다. 두 가상환경을 서브프로세스로 묶는다.

  roi-all : .venv-olkavs-roi  모든 클립 → 입술 프레임 NPZ (실패 사유 기록)
  score   : .venv-olkavs      모델 1회 로드 → CTC/hybrid 텍스트, 지정 20문장 후보 점수, 집계표
  run     : 위 둘을 순서대로 실행

사용 예:
  python scripts/pilot_evaluate.py run --clips D:/silent_talk_webcam/pilot --out experiments/webcam_pilot_20261022 --device cpu

보고 지표(조건별·문장 묶음별): NFC 문자 CER, 완전 일치, 지정 문장 후보 top-1(길이만 보는 분류기와 비교),
무발화 오수락률, 입술 검출 성공률, 처리 시간. 자유 문장은 .json 에 `spoken`(실제 말한 문장)이 있을 때만 CER을 낸다.
이 수치는 튜닝 참가자 파일럿이며 최종 평가가 아니다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import olkavs_transcribe as ot  # noqa: E402

VIDEO_SUFFIXES = {".webm", ".mp4"}
PROMPTS_PATH = ROOT / "frontend" / "src" / "capture" / "prompts.json"


def load_clips(clips_dir: Path) -> list[dict[str, Any]]:
    """영상과 짝이 맞는 메타 JSON을 읽어 클립 목록을 만든다."""
    rows = []
    for video in sorted(clips_dir.iterdir()):
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        meta_path = video.with_suffix(".json")
        if not meta_path.exists():
            rows.append({"file": video.name, "status": "missing_meta"})
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rows.append({"file": video.name, "status": "bad_meta", "error": str(exc)})
            continue
        rows.append({"file": video.name, "path": str(video), "status": "ok", "participant": meta.get("participant"),
                     "condition": meta.get("condition"), "sentence_id": meta.get("sentenceId"), "kind": meta.get("kind"),
                     "rep": meta.get("rep"), "text": meta.get("text"), "spoken": meta.get("spoken"),
                     "duration_ms": meta.get("durationMs"), "frame_callbacks": meta.get("frameCallbacks"),
                     "estimated_fps": meta.get("estimatedFps"), "track": meta.get("trackSettings")})
    return rows


def cmd_roi_all(args: argparse.Namespace) -> int:
    """모든 클립의 입술 프레임을 추출한다 (.venv-olkavs-roi)."""
    clips_dir, out = Path(args.clips).resolve(), Path(args.out).resolve()
    (out / "roi").mkdir(parents=True, exist_ok=True)
    rows = load_clips(clips_dir)
    results = []
    for row in rows:
        if row["status"] != "ok":
            results.append(row)
            continue
        stem = Path(row["file"]).stem
        clip_out = out / "roi" / stem
        ns = argparse.Namespace(video=row["path"], out=str(clip_out), start=None, end=None, resample=False)
        t0 = time.perf_counter()
        rc = ot.run_roi(ns)
        info = json.loads((clip_out / "roi.json").read_text(encoding="utf-8"))
        row = {**row, "roi_rc": rc, "roi_seconds": time.perf_counter() - t0, "roi_dir": str(clip_out)}
        if rc == 0:
            row.update(frames=info["selected_frames"]["count"], roi_status="ok",
                       detector=info["detector_stats"], video_info=info["video"], selection=info["selected_frames"])
        else:
            row.update(roi_status="failed", roi_error=info.get("error"))
        results.append(row)
        print(json.dumps({"file": row["file"], "roi": row.get("roi_status"), "frames": row.get("frames"),
                          "error": (row.get("roi_error") or {}).get("message")}, ensure_ascii=False), flush=True)
    ot.save_json(out / "roi_index.json", {"clips_dir": str(clips_dir), "n": len(results), "rows": results})
    return 0


def text_to_olkavs_tokens(text: str, bundle: dict[str, Any]) -> list[int]:
    """문장 → OLKAVS 72토큰 ID (char2grp 자모 분해 + 사전). 사전에 없는 글자는 ValueError."""
    sys.path.insert(0, str(ot.UPSTREAM))
    from avsr.utils.korean_funcs import char2grp
    import unicodedata
    vocab = bundle["vocab"].vocab_dict
    tokens = []
    for g in char2grp(" ".join(unicodedata.normalize("NFC", text).split())):
        if g not in vocab:
            raise ValueError(f"사전에 없는 토큰: {g!r}")
        tokens.append(int(vocab[g]))
    return tokens


ctc_forced_score = ot.ctc_forced_score  # 후보 문장 CTC 강제 정렬 점수 (olkavs_transcribe 공용 구현)


def cmd_score(args: argparse.Namespace) -> int:
    """입술 프레임을 인식하고 지정 문장 후보 점수·집계표를 만든다 (.venv-olkavs)."""
    import numpy as np
    import torch
    sys.path.insert(0, str(ot.OLKAVS))
    from evaluation_metrics import counts, aggregate
    out = Path(args.out).resolve()
    index = json.loads((out / "roi_index.json").read_text(encoding="utf-8"))
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    fixed = prompts["fixed"]
    device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    bundle = ot.load_olkavs(device)
    model = bundle["model"]
    cand_tokens = {p["id"]: text_to_olkavs_tokens(p["text"], bundle) for p in fixed}
    records = []
    for row in index["rows"]:
        rec = dict(row)
        if row.get("roi_status") != "ok":
            records.append(rec)
            continue
        frames = ot.read_frames(Path(row["roi_dir"]) / "roi.npz")
        res = ot.transcribe_frames(frames, device, ("ctc_greedy", "hybrid"))
        rec["outputs"] = res["outputs"]
        rec["timing"] = res["timing_seconds"]
        # CTC 로그확률로 지정 문장 후보 점수
        video, _ = ot.normalize_frames(frames)
        with torch.inference_mode():
            feats = model.medium(model.encoder(video.to(device), torch.tensor([video.shape[2]], device=device), None, None))
            logits = model.decoder.ctcdecoder(inputs=feats)[0].double().cpu()
        lp = torch.log_softmax(logits, dim=-1)
        scores = {cid: ctc_forced_score(lp, toks, blank=3) for cid, toks in cand_tokens.items()}
        feasible = {k: v for k, v in scores.items() if v["status"] == "ok"}
        ranked = sorted(feasible, key=lambda k: feasible[k]["per_token"], reverse=True)
        ranked_total = sorted(feasible, key=lambda k: feasible[k]["total"], reverse=True)
        # 길이만 보는 기준선: 프레임 수와 후보 토큰 수의 비율이 학습 데이터 평균(약 2.6프레임/토큰, 30fps)에 가장 가까운 후보
        n = len(frames)
        length_pick = min(cand_tokens, key=lambda k: abs(n / max(1, len(cand_tokens[k])) - args.frames_per_token))
        rec["candidates"] = {"ranked_per_token": ranked[:5], "ranked_total": ranked_total[:5], "length_only_pick": length_pick,
                             "margin_per_token": (feasible[ranked[0]]["per_token"] - feasible[ranked[1]]["per_token"]) if len(ranked) > 1 else None,
                             "scores": scores}
        ref = row.get("text") if row.get("kind") == "fixed" else row.get("spoken")
        if ref:
            for m, o in rec["outputs"].items():
                o["metrics"] = counts(ref, o.get("text", ""))
        records.append(rec)
        print(json.dumps({"file": row["file"], "cond": row.get("condition"), "ref": ref,
                          "hybrid": rec["outputs"]["hybrid"].get("text"), "ctc": rec["outputs"]["ctc_greedy"].get("text"),
                          "top1": ranked[0] if ranked else None}, ensure_ascii=False), flush=True)

    summary = summarize(records, fixed)
    summary.update({"device": device, "torch": torch.__version__, "checkpoint_sha256": ot.CHECKPOINT_SHA256,
                    "prompts_version": prompts.get("version"), "frames_per_token_baseline": args.frames_per_token,
                    "note": "튜닝 참가자 파일럿. AI Hub 문장(A)은 OLKAVS 학습 문구일 수 있으므로 N과 나눠 본다. 최종 평가 아님."})
    ot.save_json(out / "pilot_results.json", {"summary": summary, "records": records})
    ot.save_json(out / "pilot_summary.json", summary)
    print(json.dumps(summary["tables"], ensure_ascii=False, indent=1))
    return 0


def summarize(records: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> dict[str, Any]:
    """조건·문장 묶음별 CER/정확/후보 top-1/오수락/검출률 표."""
    sys.path.insert(0, str(ot.OLKAVS))
    from evaluation_metrics import aggregate
    subset = {p["id"]: ("aihub" if p["id"].startswith("A") else "new") for p in fixed}
    ok = [r for r in records if r.get("roi_status") == "ok" and "outputs" in r]
    tables: dict[str, Any] = {}
    groups: dict[str, list] = defaultdict(list)
    for r in ok:
        if r.get("kind") == "fixed":
            groups[f"fixed/{r['condition']}/all"].append(r)
            groups[f"fixed/{r['condition']}/{subset.get(r['sentence_id'], '?')}"].append(r)
        elif r.get("kind") == "free" and r.get("spoken"):
            groups[f"free/{r['condition']}"].append(r)
    for key, rows in sorted(groups.items()):
        entry: dict[str, Any] = {"clips": len(rows)}
        for m in ("hybrid", "ctc_greedy"):
            scored = [{"id": r["file"], "status": "ok", "metrics": r["outputs"][m]["metrics"]} for r in rows if "metrics" in r["outputs"][m]]
            if scored:
                a = aggregate(scored)
                entry[m] = {"cer_percent": round(a["cer_percent"], 2), "exact": a["exact_sentences"], "n": len(scored)}
        if key.startswith("fixed"):
            top1 = sum(1 for r in rows if r["candidates"]["ranked_per_token"][:1] == [r["sentence_id"]])
            top1_total = sum(1 for r in rows if r["candidates"]["ranked_total"][:1] == [r["sentence_id"]])
            top3 = sum(1 for r in rows if r["sentence_id"] in r["candidates"]["ranked_per_token"][:3])
            length = sum(1 for r in rows if r["candidates"]["length_only_pick"] == r["sentence_id"])
            entry["candidate_top1_per_token"] = f"{top1}/{len(rows)}"
            entry["candidate_top1_total"] = f"{top1_total}/{len(rows)}"
            entry["candidate_top3_per_token"] = f"{top3}/{len(rows)}"
            entry["length_only_top1"] = f"{length}/{len(rows)}"
            entry["chance"] = f"1/{len(fixed)}"
        tables[key] = entry
    non = [r for r in records if r.get("kind") == "nonspeech"]
    non_ok = [r for r in non if r.get("roi_status") == "ok" and "outputs" in r]
    tables["nonspeech"] = {"clips": len(non), "lip_detected": len(non_ok),
                           "hybrid_nonempty": sum(1 for r in non_ok if r["outputs"]["hybrid"].get("text")),
                           "ctc_nonempty": sum(1 for r in non_ok if r["outputs"]["ctc_greedy"].get("text"))}
    det_total = [r for r in records if r.get("status") == "ok"]
    tables["detection"] = {"clips": len(det_total), "roi_ok": sum(1 for r in det_total if r.get("roi_status") == "ok"),
                           "failures": sorted({(r.get("roi_error") or {}).get("message", "")[:60] for r in det_total if r.get("roi_status") != "ok"})}
    times = [r["timing"]["encoder"] + r["outputs"]["hybrid"].get("seconds", 0) for r in ok if "timing" in r]
    tables["timing_seconds"] = {"encoder_plus_hybrid_median": sorted(times)[len(times) // 2] if times else None,
                                "roi_median": sorted(r["roi_seconds"] for r in ok)[len(ok) // 2] if ok else None}
    return {"tables": tables, "clips_total": len(records)}


def cmd_run(args: argparse.Namespace) -> int:
    """roi-all → score 를 각 가상환경으로 실행한다."""
    script = str(Path(__file__).resolve())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rc = subprocess.run([str(ot.ROI_PYTHON), script, "roi-all", "--clips", args.clips, "--out", args.out], env=env).returncode
    if rc != 0:
        return rc
    return subprocess.run([str(ot.INFER_PYTHON), script, "score", "--out", args.out, "--device", args.device,
                           "--threads", str(args.threads), "--frames-per-token", str(args.frames_per_token)], env=env).returncode


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("roi-all"); p.add_argument("--clips", required=True); p.add_argument("--out", required=True); p.set_defaults(func=cmd_roi_all)
    p = sub.add_parser("score"); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4); p.add_argument("--frames-per-token", type=float, default=2.6); p.set_defaults(func=cmd_score)
    p = sub.add_parser("run"); p.add_argument("--clips", required=True); p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--frames-per-token", type=float, default=2.6); p.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
