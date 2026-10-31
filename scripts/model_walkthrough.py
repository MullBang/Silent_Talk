"""영상 하나가 서비스 인식 경로를 지나가는 과정을 단계별로 보여 주는 HTML을 만든다 (내부 확인용).

녹화 화면(`POST /api/record`)과 같은 경로를 쓴다: 입술 찾기 → 입 움직임으로 말한 구간 자르기 → 96x96 입술 프레임 →
OLKAVS 프레임별 자모 확률 → CTC·hybrid 글자 → 신뢰도 판정. 결과 HTML 하나에 영상과 얼굴 이미지가 들어가므로
`--out`은 저장소 밖 폴더로 주고, 외부에 게시하지 않는다.

하위 명령
  run    : 아래 셋을 순서대로 실행한다 (아무 Python)
  roi    : .venv-olkavs-roi  영상 → 입술 창 NPZ, 프레임별 입 벌림 비율·입술 상자, 대표 이미지
  infer  : .venv-olkavs      입술 창 → 프레임별 상위 자모, CTC·hybrid 글자, 신뢰도
  render : 아무 Python       위 결과 → index.html

사용 예
  py -3.11 scripts/model_walkthrough.py run --video clip.mp4 --out C:/.../walkthrough/clip --reference "정답 문장"
  py -3.11 scripts/model_walkthrough.py run --lip-video aihub_96.mp4 --out C:/.../walkthrough/aihub --reference "정답 문장"
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import olkavs_transcribe as ot  # noqa: E402

MIN_CONFIDENCE = 0.75     # backend/config.py OLKAVS_MIN_CONFIDENCE와 같은 값
FILM_FRAMES = 24          # 입술 프레임 필름에 보여 줄 장 수
BLANK_ID = 3              # OLKAVS 사전의 unk = CTC 빈칸
TOP_K = 3


def run_roi(args: argparse.Namespace) -> int:
    """서비스 녹화 경로(extract_segments utterance=True)를 돌리며 프레임별 값을 함께 기록한다 (.venv-olkavs-roi)."""
    import cv2
    import numpy as np
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.lip_video:
            cap = cv2.VideoCapture(str(args.lip_video))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            crops = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                crops.append(frame)
            cap.release()
            if not crops:
                raise ValueError(f"입술 영상을 읽지 못했습니다: {args.lip_video}")
            frames = np.stack(crops).astype(np.uint8)
            npz = out / "seg_00.npz"
            np.savez_compressed(npz, frames=frames)
            ms = int(len(crops) / fps * 1000)
            save_film(frames, out)
            ot.save_json(out / "roi_steps.json", {"mode": "lip_video", "rate": fps, "frames_log": [], "segments": [
                {"index": 0, "start_ms": 0, "end_ms": ms, "frames": len(crops), "status": "ok", "npz": str(npz)}]})
            return 0
        log: list[dict[str, Any]] = []
        best: dict[str, Any] = {"ratio": -1.0, "frame": None, "box": None}
        base = ot.vad_tracker_class()

        class SpyTracker(base):  # type: ignore[misc, valid-type]
            """서비스 추적기와 같고, 프레임마다 상태·상자·입 벌림 비율을 기록만 한다."""

            def process(self, frame_bgr: Any) -> Any:
                """부모 처리 후 값을 복사해 둔다. 입이 가장 크게 벌어진 프레임을 대표 이미지로 남긴다."""
                roi = super().process(frame_bgr)
                record = self.frame_records[-1] if self.frame_records else {}
                box = getattr(self, "last_box_xyxy", None)
                ratio = self.mouth_ratio
                log.append({"status": record.get("status"), "ratio": ratio,
                            "box": [int(v) for v in box] if box is not None else None})
                if ratio is not None and box is not None and ratio > best["ratio"]:
                    best.update(ratio=ratio, frame=frame_bgr.copy(), box=[int(v) for v in box])
                return roi

        result = ot.extract_segments(Path(args.video).resolve(), out, utterance=True, tracker_factory=SpyTracker,
                                     geometry_factory=_geometry_factory())
        ok_segments = [s for s in result["segments"] if s.get("status") == "ok" and s.get("npz")]
        for seg in ok_segments:
            with np.load(seg["npz"], allow_pickle=False) as data:
                frames = data["frames"]
            save_film(frames, out)
            break
        if best["frame"] is not None:
            frame, (x1, y1, x2, y2) = best["frame"], best["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), max(2, frame.shape[1] // 400))
            scale = 720 / frame.shape[1]
            small = cv2.resize(frame, (720, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "sample.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rate = float(result["selection"].get("rate") or 30.0)
        ratios = [r["ratio"] for r in log]
        ot.save_json(out / "roi_steps.json", {
            "mode": "video", "rate": rate, "frames_log": log, "active": [bool(a) for a in ot.speech_activity(ratios, rate)],
            "segments": result["segments"], "speech": result.get("speech"), "vad": result.get("vad"),
            "selection": result.get("selection"), "stats": result.get("stats"), "timing_seconds": result.get("timing_seconds")})
        return 0
    except Exception as exc:  # noqa: BLE001 - 명령줄 도구의 최종 보고
        ot.save_json(out / "roi_steps.json", {"mode": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[roi] 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _geometry_factory() -> Any:
    """서비스와 같은 입술 정사각형 계산기를 만드는 함수."""
    ot._ensure_path(ot.OLKAVS)
    from roi_geometry_step04 import RoiGeometry
    return lambda: RoiGeometry(ot.GEOMETRY_ARM)


def save_film(frames: Any, out: Path) -> None:
    """입술 프레임을 고르게 FILM_FRAMES장 뽑아 jpg로 저장한다."""
    import cv2
    count = len(frames)
    picks = sorted({round(i * (count - 1) / max(1, FILM_FRAMES - 1)) for i in range(FILM_FRAMES)})
    film = out / "film"
    film.mkdir(exist_ok=True)
    for name in film.glob("*.jpg"):
        name.unlink()
    for rank, idx in enumerate(picks):
        cv2.imwrite(str(film / f"{rank:02d}_{idx:04d}.jpg"), frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 90])


def run_infer(args: argparse.Namespace) -> int:
    """입술 창마다 서비스 recognize와 같은 모델로 프레임별 상위 자모와 글자를 구한다 (.venv-olkavs)."""
    import numpy as np
    import torch
    out = Path(args.out).resolve()
    try:
        steps = json.loads((out / "roi_steps.json").read_text(encoding="utf-8"))
        device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        bundle = ot.load_olkavs(device)
        model, vocab = bundle["model"], bundle["vocab"]
        windows = []
        for seg in steps.get("segments", []):
            if seg.get("status") != "ok" or not seg.get("npz"):
                continue
            with np.load(seg["npz"], allow_pickle=False) as data:
                frames = data["frames"]
            hybrid = ot.recognize(frames, device, "hybrid")
            ctc = ot.recognize(frames, device, "ctc")
            video, _ = ot.normalize_frames(frames)
            video = video.to(device)
            with torch.inference_mode():
                features = model.medium(model.encoder(video, torch.tensor([video.shape[2]], device=device), None, None))
                log_probs = torch.log_softmax(model.decoder.ctcdecoder(inputs=features)[0].double(), dim=-1)
            top = torch.topk(log_probs.exp(), TOP_K, dim=-1)
            per_frame = [[{"id": int(i), "token": token_name(vocab, int(i)), "p": round(float(p), 4)}
                          for i, p in zip(ids.tolist(), probs.tolist())]
                         for ids, probs in zip(top.indices.cpu(), top.values.cpu())]
            windows.append({"start_ms": seg.get("start_ms"), "end_ms": seg.get("end_ms"), "frames": int(frames.shape[0]),
                            "input_shape": [1, 3, int(frames.shape[0]), 96, 96], "feature_shape": list(features.shape),
                            "hybrid": {k: hybrid[k] for k in ("text", "confidence", "raw_score", "timing_seconds")},
                            "ctc": {k: ctc[k] for k in ("text", "confidence", "timing_seconds")},
                            "ctc_jamo": " ".join(token_name(vocab, t) for t in ctc["token_ids"]), "per_frame": per_frame})
        ot.save_json(out / "model.json", {"device": device, "windows": windows})
        return 0
    except Exception as exc:  # noqa: BLE001 - 명령줄 도구의 최종 보고
        print(f"[infer] 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def token_name(vocab: Any, token_id: int) -> str:
    """토큰 번호 → 보여 줄 글자. CTC 빈칸은 '·', 공백은 '␣'."""
    if token_id == BLANK_ID:
        return "·"
    name = str(vocab.id_dict.get(token_id, "?"))
    return "␣" if name == " " else name


def token_html(token: str) -> str:
    """보여 줄 자모 HTML. OLKAVS 사전에서 역슬래시가 붙은 자음은 받침이므로 밑줄로 표시한다."""
    if token.startswith("\\"):
        return f"<u>{html.escape(token[1:])}</u>"
    return html.escape(token)


def data_uri(path: Path, mime: str) -> str:
    """파일을 HTML 안에 넣을 data URI로 바꾼다."""
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def char_error(reference: str, text: str) -> tuple[int, int]:
    """공백을 뺀 글자 기준 편집 거리와 정답 글자 수."""
    a, b = reference.replace(" ", ""), text.replace(" ", "")
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1], len(a)


def ratio_chart(steps: dict[str, Any], kept: list[tuple[float, float]]) -> str:
    """입 벌림 비율 시계열 SVG: 초록 띠 = 모델에 넘긴 구간, 파란 선 = 입 벌림 비율."""
    log = steps.get("frames_log") or []
    rate = float(steps.get("rate") or 30.0)
    values = [(i / rate, r["ratio"]) for i, r in enumerate(log) if r.get("ratio") is not None]
    if len(values) < 2:
        return "<p>입 벌림 비율을 구하지 못했습니다.</p>"
    width, height, pad = 860.0, 170.0, 28.0
    t_max = max(len(log) / rate, 0.001)
    lo, hi = min(v for _, v in values), max(v for _, v in values)
    span = (hi - lo) or 1.0
    x = lambda t: pad + (width - 2 * pad) * t / t_max  # noqa: E731
    y = lambda v: height - pad - (height - 2 * pad) * (v - lo) / span  # noqa: E731
    bands = "".join(f'<rect x="{x(a):.1f}" y="{pad / 2:.1f}" width="{max(1.0, x(b) - x(a)):.1f}" height="{height - 1.5 * pad:.1f}" fill="var(--band)"/>'
                    for a, b in kept)
    line = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in values)
    ticks = "".join(f'<text x="{x(s):.1f}" y="{height - 8:.1f}" text-anchor="middle" class="tick">{s}초</text>'
                    for s in range(0, int(t_max) + 1))
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="시간에 따른 입 벌림 비율">{bands}'
            f'<polyline points="{line}" fill="none" stroke="var(--accent)" stroke-width="2"/>{ticks}</svg>')


def frame_strip(window: dict[str, Any], rate: float) -> str:
    """프레임별 1순위 자모 띠. 진할수록 모델이 확신한 것이고, 올리면 상위 3개가 보인다."""
    cells = []
    for i, tops in enumerate(window["per_frame"]):
        best = tops[0]
        tip = f"{i / rate:.2f}초: " + ", ".join(f"{t['token'].lstrip(chr(92))} {t['p'] * 100:.0f}%" for t in tops)
        kind = "blank" if best["id"] == BLANK_ID else "tok"
        alpha = 0.12 + 0.75 * best["p"]
        cells.append(f'<span class="cell {kind}" style="--a:{alpha:.2f}" title="{html.escape(tip)}">{token_html(best["token"])}</span>')
    return f'<div class="strip">{"".join(cells)}</div>'


def render(args: argparse.Namespace) -> int:
    """roi_steps.json·model.json·이미지로 index.html 하나를 만든다."""
    out = Path(args.out).resolve()
    try:
        steps = json.loads((out / "roi_steps.json").read_text(encoding="utf-8"))
        model = json.loads((out / "model.json").read_text(encoding="utf-8"))
        windows = model["windows"]
        rate = float(steps.get("rate") or 30.0)
        film = "".join(f'<img src="{data_uri(p, "image/jpeg")}" alt="입술 프레임 {p.stem[3:]}" width="72" height="72">'
                       for p in sorted((out / "film").glob("*.jpg")))
        parts: list[str] = [f"<h1>{html.escape(args.title or '모델 작동 과정')}</h1>"]
        parts.append('<p class="lead">아래는 이 영상 하나가 실제 서비스와 같은 경로를 지나간 기록입니다. 소리는 쓰지 않습니다.</p>')
        if args.video and steps.get("mode") == "video":
            suffix = Path(args.video).suffix.lower()
            mime = "video/webm" if suffix == ".webm" else "video/mp4"
            parts.append(f'<section><h2>1. 입력 영상</h2><video controls muted src="{data_uri(Path(args.video), mime)}"></video>'
                         f'<p>모델은 이 영상의 소리를 쓰지 않습니다. 화면만 봅니다. 초당 {rate:.1f}프레임으로 읽었습니다.</p></section>')
        if (out / "sample.jpg").exists():
            parts.append(f'<section><h2>2. 입술 찾기</h2><img class="wide" src="{data_uri(out / "sample.jpg", "image/jpeg")}" alt="입술 상자를 표시한 프레임">'
                         '<p>프레임마다 얼굴 랜드마크(MediaPipe)로 입술 위치를 찾습니다. 초록 상자가 찾은 입술입니다. '
                         '그 주변을 흔들리지 않는 정사각형으로 잘라 96×96 크기로 줄입니다.</p></section>')
        kept = [(w["start_ms"] / 1000, w["end_ms"] / 1000) for w in windows]
        if steps.get("mode") == "video":
            parts.append(f'<section><h2>3. 말한 구간 고르기</h2>{ratio_chart(steps, kept)}'
                         '<p>파란 선은 입이 벌어진 정도입니다. 이 선이 움직이는 곳을 말하는 구간으로 보고, 초록 띠 부분만 모델에 넘깁니다. '
                         '입이 전혀 안 움직이면 여기서 [발화 없음]으로 끝납니다.</p></section>')
        parts.append(f'<section><h2>{"4" if steps.get("mode") == "video" else "1"}. 모델에 들어가는 입술 프레임</h2><div class="film">{film}</div>'
                     f'<p>모델이 실제로 보는 전부입니다. 고르게 뽑은 일부만 표시했습니다.</p></section>')
        for n, w in enumerate(windows):
            label = f" (구간 {n + 1})" if len(windows) > 1 else ""
            conf = float(w["hybrid"]["confidence"])
            passed = conf >= args.min_confidence
            shown = w["hybrid"]["text"] if passed else "[인식 불가]"
            verdict = f"기준 {args.min_confidence:.2f} 이상이라 그대로 보여 줍니다." if passed else f"기준 {args.min_confidence:.2f}보다 낮아 [인식 불가]로 보여 줍니다."
            num = "5" if steps.get("mode") == "video" else "2"
            parts.append(f'<section><h2>{num}. 모델 출력{label}: 프레임마다 자모 하나</h2>{frame_strip(w, rate)}'
                         f'<p>입술 프레임 {w["frames"]}장을 넣으면 시간 순서를 유지한 특징 {w["feature_shape"][1]}칸이 나오고, 칸마다 자모 72종 중 무엇인지 확률을 냅니다. '
                         '위 띠의 글자 하나가 프레임 하나입니다. 진할수록 모델이 확신한 것이고, 가운뎃점(·)은 "이 칸에는 새 글자 없음", ␣는 띄어쓰기, 밑줄은 받침입니다. 마우스를 올리면 상위 3개가 보입니다.</p>'
                         '</section>')
            num2 = "6" if steps.get("mode") == "video" else "3"
            rows = [f'<tr><th>CTC 자모</th><td class="jamo">{" ".join(token_html(t) for t in w["ctc_jamo"].split(" "))}</td></tr>',
                    f'<tr><th>CTC 글자</th><td>{html.escape(w["ctc"]["text"])}</td></tr>',
                    f'<tr><th>hybrid 글자</th><td>{html.escape(w["hybrid"]["text"])}</td></tr>']
            if args.reference:
                edits, chars = char_error(args.reference, w["hybrid"]["text"])
                rows.append(f'<tr><th>{html.escape(args.reference_label)}</th><td>{html.escape(args.reference)}</td></tr>')
                rows.append(f'<tr><th>틀린 글자 비율</th><td>{edits / max(1, chars) * 100:.1f}% ({edits}/{chars}, 공백 제외)</td></tr>')
            rows.append(f'<tr><th>신뢰도</th><td>{conf:.3f}. {html.escape(verdict)}</td></tr>')
            rows.append(f'<tr class="final"><th>화면에 나오는 결과</th><td>{html.escape(shown)}</td></tr>')
            parts.append(f'<section><h2>{num2}. 자모를 글자로{label}</h2><table>{"".join(rows)}</table>'
                         '<p><b>CTC</b>는 띠에서 이어진 같은 자모를 하나로 합치고 빈칸을 지운 결과입니다. 입 모양만 따릅니다. '
                         '<b>hybrid</b>는 CTC 점수(60%)와, 앞 글자를 보고 다음 글자를 예측하는 디코더 점수(40%)를 섞어 문장을 고릅니다. '
                         '서비스는 hybrid를 씁니다. 그래서 입 모양이 불분명해도 자연스러운 문장처럼 나올 수 있고, 그럴듯해 보여도 틀릴 수 있습니다.</p>'
                         f'<p class="small">처리 시간: 인코더 {w["hybrid"]["timing_seconds"].get("encoder", 0):.2f}초, 디코딩 {w["hybrid"]["timing_seconds"].get("decode", 0):.2f}초 ({html.escape(model["device"])})</p></section>')
        if not windows:
            parts.append("<section><h2>결과</h2><p>모델에 넘길 구간이 없었습니다(얼굴을 못 찾았거나 말한 구간이 없음).</p></section>")
        page = PAGE.replace("{{BODY}}", "\n".join(parts)).replace("{{TITLE}}", html.escape(args.title or "모델 작동 과정"))
        (out / "index.html").write_text(page, encoding="utf-8")
        print(out / "index.html")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[render] 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root { --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#fff; --soft:#f3f4f6; --accent:#2563eb; --band:rgba(22,163,74,.18); --tok:37,99,235; }
@media (prefers-color-scheme: dark) { :root { --fg:#e5e7eb; --muted:#9ca3af; --line:#374151; --bg:#111827; --soft:#1f2937; --accent:#60a5fa; --band:rgba(74,222,128,.18); --tok:96,165,250; } }
body { margin:0 auto; max-width:920px; padding:24px 16px 64px; background:var(--bg); color:var(--fg); font-family:'Malgun Gothic','Apple SD Gothic Neo',system-ui,sans-serif; line-height:1.65; }
h1 { font-size:26px; margin-bottom:4px; } h2 { font-size:19px; margin:0 0 10px; } .lead { color:var(--muted); margin-top:0; }
section { border-top:1px solid var(--line); padding:20px 0; } p { margin:10px 0 0; } .small { color:var(--muted); font-size:13px; }
video, img.wide { width:100%; max-width:720px; border-radius:8px; display:block; background:#000; }
.film { display:flex; flex-wrap:wrap; gap:4px; } .film img { border-radius:4px; }
svg { width:100%; height:auto; display:block; } .tick { font-size:11px; fill:var(--muted); }
.strip { display:flex; flex-wrap:wrap; gap:1px; }
.cell { flex:0 0 22px; height:30px; line-height:30px; text-align:center; font-size:14px; border-radius:3px; }
.cell.tok { background:rgba(var(--tok),var(--a)); } .cell.blank { background:var(--soft); color:var(--muted); }
table { border-collapse:collapse; width:100%; } th, td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
th { width:150px; background:var(--soft); font-weight:600; } .jamo { font-size:13px; word-break:break-all; } .final td { font-size:19px; font-weight:700; }
</style></head><body>
{{BODY}}
</body></html>
"""


def run_all(args: argparse.Namespace) -> int:
    """roi → infer → render 를 각 가상환경으로 실행한다."""
    script = str(Path(__file__).resolve())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = Path(args.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    source = ["--lip-video", args.lip_video] if args.lip_video else ["--video", args.video]
    rc = subprocess.run([str(ot.ROI_PYTHON), script, "roi", *source, "--out", str(out)], env=env).returncode
    if rc != 0:
        return rc
    rc = subprocess.run([str(ot.INFER_PYTHON), script, "infer", "--out", str(out), "--device", args.device], env=env).returncode
    if rc != 0:
        return rc
    return render(args)


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--video", default=None, help="원본 영상 (얼굴이 보이는 mp4/webm)")
        p.add_argument("--lip-video", default=None, help="이미 잘린 96x96 입술 영상 (입술 찾기 단계를 건너뜀)")
        p.add_argument("--out", required=True, help="저장소 밖 폴더")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--title", default=None)
        p.add_argument("--reference", default=None, help="정답 문장 (있으면 틀린 글자 비율을 보여 줌)")
        p.add_argument("--reference-label", default="정답")
        p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)

    for name, func in (("run", run_all), ("roi", run_roi), ("infer", run_infer), ("render", render)):
        p = sub.add_parser(name)
        common(p)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    if args.command in ("run", "roi") and not (args.video or args.lip_video):
        parser.error("--video 또는 --lip-video가 필요합니다")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
