"""train의 소수 문장만 암기시켜 학습/CTC/한글 복원 연결을 확인한다.

val/test 추론이나 일반화 평가를 하지 않는다. 새 빈 결과 폴더에만 저장한다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train import collate, ctc_training_loss, resolve_device_precision
from models.baseline import ARCHITECTURE_VERSION, LipNetBaseline
from models.registry import ARCHITECTURES, create_model, load_model_checkpoint
from models.cache import load_clip
from models.dataset_manifest import DEFAULT_MANIFEST, load_manifest, sha256_file
from models.decoder import ctc_greedy_decode, decode
from models.tokenizer import cer, wer


def select_records(document: dict, count: int) -> list[dict]:
    """예측을 보기 전에 train 중 길이가 짧은 순서로 고정 선정한다."""
    rows = sorted((r for r in document["records"] if r["split"] == "train"),
                  key=lambda r: (r["frames"], r["id"]))
    if count < 2 or count > len(rows):
        raise ValueError("표본 수는 2 이상이고 train 문장 수 이하여야 합니다.")
    selected = rows[:count]
    if len({r["text"] for r in selected}) != count:
        raise ValueError("같은 정답이 반복되는 표본입니다. 별도 선정 정책이 필요합니다.")
    return selected


@torch.no_grad()
def evaluate(model: LipNetBaseline, items: list[dict], device: torch.device,
             *, beam: bool = False, control: str | None = None) -> dict:
    """각 영상을 개별 추론한다. 정답은 출력 이후 비교에만 사용한다."""
    model.eval()
    rows = []
    for item in items:
        example = item
        if control:
            frames = item["frames"]
            if control == "reversed":
                frames = frames[::-1].copy()
            elif control == "constant_gray":
                frames = np.full_like(frames, 128)
            else:
                raise ValueError(f"알 수 없는 대조 입력: {control}")
            example = {**item, "frames": frames}
        x, _, _, _ = collate([example])
        # 학습 loss가 아니라 실제 추론 경로(개별 입력, lengths 없이 FP32)를 검사한다.
        log_probs = model(x.to(device)).cpu()
        greedy = ctc_greedy_decode(log_probs)
        row = {"reference": item["text"], "greedy": greedy,
               "greedy_cer": cer(item["text"], greedy), "greedy_wer": wer(item["text"], greedy),
               "greedy_exact": greedy == item["text"]}
        if beam:
            predicted, confidence, raw_score = decode(log_probs)
            row.update({"beam": predicted, "beam_cer": cer(item["text"], predicted),
                        "beam_wer": wer(item["text"], predicted), "beam_exact": predicted == item["text"],
                        "confidence": confidence, "raw_score": raw_score})
        rows.append(row)
    result = {"samples": rows, "mean_greedy_cer": sum(r["greedy_cer"] for r in rows) / len(rows),
              "greedy_exact_count": sum(r["greedy_exact"] for r in rows)}
    if beam:
        result.update({"mean_beam_cer": sum(r["beam_cer"] for r in rows) / len(rows),
                       "beam_exact_count": sum(r["beam_exact"] for r in rows)})
    return result


def run(args: argparse.Namespace) -> dict:
    """사전 지정한 성공 기준까지 반복하고, 재로드/대조 결과와 출처를 저장한다."""
    if min(args.max_steps, args.eval_every, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("steps/eval/batch/lr은 양수여야 합니다.")
    out = Path(args.out_dir).resolve()
    # 실험 결과가 배포 가중치/원본 데이터와 섞이지 않게 새 디렉터리만 허용한다.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"결과를 덮어쓰지 않습니다. 새 빈 폴더를 지정하세요: {out}")
    document, summary = load_manifest(args.manifest)
    selected = select_records(document, args.samples)
    base = Path(args.manifest).resolve().parent
    items = [load_clip(base / row["cache_path"]) for row in selected]
    out.mkdir(parents=True, exist_ok=True)
    device, use_amp = resolve_device_precision(args.device, args.precision)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    architecture = getattr(args, "architecture", ARCHITECTURE_VERSION)
    model = create_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024)
    provenance = {"mode": "smoke_only", "purpose": "train_subset_overfit",
                  "generalization_evaluation": False, "manifest_sha256": summary["manifest_sha256"],
                  "sample_ids": [r["id"] for r in selected]}
    report = {
        "status": "running", "architecture_version": architecture,
        "purpose": "Training-set memorization check only; no generalization evaluation",
        "settings": vars(args), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_data": provenance,
        "selection_policy": "Shortest train clips by (frames, id), chosen before training",
        "selected": [{k: r[k] for k in ("id", "speaker_id", "split", "frames", "text", "cache_path", "cache_sha256")}
                     for r in selected],
        "success_rule": "All selected sentences exactly reproduced by greedy decoding at two consecutive checks; beam and CPU reload also exact",
        "initial": evaluate(model, items, device), "history": [],
        "optimizer_steps": 0, "skipped_optimizer_steps": 0,
    }
    (out / "selection.json").write_text(json.dumps(report["selected"], ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    consecutive = 0
    order = []
    for step in range(1, args.max_steps + 1):
        if not order:
            order = list(range(len(items)))
            random.shuffle(order)
        chosen, order = order[:args.batch_size], order[args.batch_size:]
        batch = collate([items[i] for i in chosen])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ctc_training_loss(model, batch, device, use_amp)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not use_amp)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        report["skipped_optimizer_steps"] += int(skipped)
        report["optimizer_steps"] += int(not skipped)
        if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
            evaluation = evaluate(model, items, device)
            consecutive = consecutive + 1 if evaluation["greedy_exact_count"] == len(items) else 0
            entry = {"step": step, "last_batch_loss": loss.item(),
                     "gradient_norm": norm.item() if torch.isfinite(norm) else None,
                     "elapsed_seconds": time.perf_counter() - started, **evaluation}
            report["history"].append(entry)
            report["completed_steps"] = step
            print(f"step={step} loss={loss.item():.4f} CER={evaluation['mean_greedy_cer']:.4f} "
                  f"exact={evaluation['greedy_exact_count']}/{len(items)}", flush=True)
            for row in evaluation["samples"]:
                print(f"  {row['reference']} -> {row['greedy']}", flush=True)
            model.save_checkpoint(str(out / "last.pt"), training_data=provenance,
                                  training_run={"overfit_only": True, "step": step, "settings": vars(args)})
            (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if consecutive >= 2:
                break
    report["final_gpu"] = evaluate(model, items, device, beam=True)
    checkpoint = out / "last.pt"
    loaded = load_model_checkpoint(str(checkpoint))
    report["reloaded_cpu"] = evaluate(loaded, items, torch.device("cpu"), beam=True)
    report["controls"] = {name: evaluate(model, items, device, control=name)
                          for name in ("constant_gray", "reversed")}
    report["passed"] = (consecutive >= 2 and all(
        report[key][method] == len(items) for key in ("final_gpu", "reloaded_cpu")
        for method in ("greedy_exact_count", "beam_exact_count")))
    report["status"] = "passed" if report["passed"] else "not_converged"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["checkpoint_sha256"] = sha256_file(checkpoint)
    if device.type == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{report['status']}] {out / 'result.json'}", flush=True)
    return report


def main() -> None:
    """GPU에서 기본 네 문장 암기 검증을 실행한다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=ARCHITECTURE_VERSION)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["auto", "amp", "fp32"], default="amp")
    args = parser.parse_args()
    result = run(args)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
