"""LipNet 베이스라인 CTC 학습 스크립트.

prepare_data/prepare_batch가 생성한 문장별 .npz(frames, label)를 로드해
3D-CNN+BiLSTM+CTC 모델을 학습한다. GPU가 있으면 자동 사용한다.
train/val 분리, val CER 모니터링, best 체크포인트 저장을 지원한다.

mediapipe를 import하지 않는다(정규화 인라인) → torch와의 네이티브 충돌 회피.

사용:
  python scripts/train.py --manifest data_cache/datasets/aihub_pilot_v2/manifest.json \
      --epochs 150 --batch-size 2 --lr 1e-4 --eval-every 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

# backend 루트 경로 추가
_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from config import MODEL_WEIGHTS_PATH, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from models.baseline import ARCHITECTURE_VERSION, LipNetBaseline  # noqa: E402
from models.registry import ARCHITECTURES, create_model, load_model_checkpoint  # noqa: E402
from models.decoder import decode  # noqa: E402
from models.cache import load_clip  # noqa: E402
from models.tokenizer import cer as _cer, validate_tokenizer_metadata  # noqa: E402
from models.dataset_manifest import DEFAULT_MANIFEST, load_manifest, split_files, sha256_file  # noqa: E402

_MEAN = np.array(NORMALIZE_MEAN, dtype=np.float32)
_STD = np.array(NORMALIZE_STD, dtype=np.float32)


def _normalize(frames_uint8: np.ndarray) -> np.ndarray:
    """(T,96,96,3) uint8 → ImageNet 정규화 float32."""
    return (frames_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD


def clip_length(path: str) -> int:
    """npz 클립의 프레임 수(T)를 반환한다.

    Args:
        path: 문장 클립 .npz 경로.

    Returns:
        프레임 수 T. 읽기 실패 시 0.
    """
    try:
        with np.load(path, allow_pickle=False) as d:
            return int(d["frames"].shape[0])
    except Exception:  # noqa: BLE001 - 손상 파일은 길이 0으로 제외 대상 표시
        return 0


class ClipDataset(Dataset):
    """문장별 .npz(frames, label, text) 데이터셋."""

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return load_clip(self.files[idx])


class LengthBucketBatchSampler(Sampler):
    """길이가 비슷한 클립끼리 묶는 배치 샘플러.

    가변 길이(짧은 42프레임 ~ 긴 294프레임) 클립을 무작위로 섞어 배치화하면,
    collate의 last-frame 패딩이 짧은 클립을 배치 내 최대 길이까지 늘려 계산·
    메모리를 크게 낭비한다. 길이순으로 정렬해 인접 클립끼리 배치를 구성하면
    패딩량이 최소화되어 긴 문장 학습이 안정적이다. 배치 '순서'는 매 epoch
    섞어 학습 편향을 막는다.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True) -> None:
        self._order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self._batches = [
            self._order[i:i + batch_size]
            for i in range(0, len(self._order), batch_size)
        ]
        self._shuffle = shuffle

    def __iter__(self):
        batches = list(self._batches)
        if self._shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


def collate(batch: list[dict]):
    """가변 길이 T를 last-frame 패딩으로 배치화 (zero-padding 금지)."""
    t_max = max(b["frames"].shape[0] for b in batch)
    tensors, in_lens, targets, tgt_lens = [], [], [], []
    for b in batch:
        f = _normalize(b["frames"])
        t = f.shape[0]
        if t < t_max:
            f = np.concatenate([f, np.repeat(f[-1:], t_max - t, axis=0)], axis=0)
        tensors.append(torch.from_numpy(f).permute(3, 0, 1, 2))
        in_lens.append(t)
        targets.append(torch.from_numpy(b["label"]))
        tgt_lens.append(len(b["label"]))
    return (
        torch.stack(tensors, dim=0),
        torch.cat(targets),
        torch.tensor(in_lens, dtype=torch.long),
        torch.tensor(tgt_lens, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_val(model, files: list[str], device: str) -> tuple[float, list[tuple[str, str]]]:
    """검증셋 평균 CER과 (정답, 예측) 샘플을 반환한다."""
    model.eval()
    ds = ClipDataset(files)
    cers, samples = [], []
    for i in range(len(ds)):
        item = ds[i]
        x, _t, _il, _tl = collate([item])
        log_probs = model(x.to(device)).cpu()
        pred, _conf, _raw = decode(log_probs)
        cers.append(_cer(item["text"], pred))
        if i < 3:
            samples.append((item["text"], pred))
    model.train()
    return (sum(cers) / len(cers) if cers else 1.0), samples


def training_files(args) -> tuple[list[str], list[str], dict]:
    """고정 명세를 검증해 train/val만 선택한다. test는 학습에 반환하지 않는다."""
    if args.smoke_data:
        files = sorted(glob.glob(os.path.join(args.smoke_data, "*.npz")))
        if not files:
            raise FileNotFoundError(f"스모크 데이터가 없습니다: {args.smoke_data}")
        for path in files:
            with np.load(path, allow_pickle=False) as cached:
                validate_tokenizer_metadata(cached, path)
            load_clip(path)
        if args.max_frames:
            files = [f for f in files if clip_length(f) <= args.max_frames]
        if not files:
            raise ValueError("스모크 학습에 사용할 클립이 없습니다.")
        return files, [], {"mode": "smoke_only", "generalization_evaluation": False}
    document, summary = load_manifest(args.manifest)
    if args.max_frames and any(r["frames"] > args.max_frames for r in document["records"]):
        raise ValueError("고정 명세를 학습 시 다시 필터링하지 않습니다. 길이 상한을 바꾼 새 명세를 만드세요.")
    provenance = {"mode": "speaker_disjoint", "manifest_sha256": summary["manifest_sha256"],
                  "policy": document["policy"], "splits": summary["splits"]}
    return (split_files(document, args.manifest, "train"),
            split_files(document, args.manifest, "val"), provenance)


def resolve_device_precision(device_name: str | None, precision: str) -> tuple[torch.device, bool]:
    """GPU 명시 요청은 실패 시 CPU로 숨겨 전환하지 않는다."""
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("학습 장치는 cpu 또는 cuda여야 합니다.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. .venv-gpu의 Python과 GPU 설치 상태를 확인하세요.")
    if precision not in ("auto", "fp32", "amp"):
        raise ValueError("precision은 auto/fp32/amp 중 하나여야 합니다.")
    if precision == "amp" and device.type != "cuda":
        raise ValueError("AMP 학습은 CUDA 환경에서만 지원합니다.")
    return device, device.type == "cuda" and precision != "fp32"


def ctc_training_loss(model, batch, device: torch.device, use_amp: bool) -> torch.Tensor:
    """CNN/LSTM은 선택적 FP16, CTC는 항상 FP32로 계산한다."""
    x, targets, in_lens, tgt_lens = batch
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        log_probs = model(x.to(device), in_lens)
    # 캐시 단계에서 CTC 정렬 가능성을 이미 검사했다. 불가능한 정렬/NaN은 숨기지 않는다.
    with torch.amp.autocast(device.type, enabled=False):
        loss = nn.functional.ctc_loss(
            log_probs.float(), targets.to(device), in_lens, tgt_lens,
            blank=0, zero_infinity=False,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("CTC loss가 유한하지 않습니다. 라벨/길이/정밀도를 확인하세요.")
    return loss


def initialize_model(architecture: str, init_weights: str | None, provenance: dict,
                     train_files: list[str]) -> tuple[torch.nn.Module, dict]:
    """같은 명세의 train 소수 문장 검사 가중치만 초기값으로 허용한다. 학습 재개가 아니다."""
    if not init_weights:
        return create_model(architecture), {"initialization": "random"}
    model = load_model_checkpoint(init_weights)
    if model.architecture_version != architecture:
        raise ValueError("초기 가중치와 요청한 모델 구조가 다릅니다.")
    source = model.training_data or {}
    source_run = model.training_run or {}
    if not isinstance(source, dict) or not isinstance(source_run, dict):
        raise ValueError("초기 가중치의 학습 출처가 올바른 객체가 아닙니다.")
    ids = source.get("sample_ids")
    train_ids = {Path(p).stem for p in train_files}
    if (provenance.get("mode") != "speaker_disjoint"
            or source.get("mode") != "smoke_only"
            or source.get("purpose") != "train_subset_overfit"
            or source.get("manifest_sha256") != provenance.get("manifest_sha256")
            or not provenance.get("manifest_sha256")
            or source_run.get("overfit_only") is not True
            or not isinstance(ids, list) or not ids
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids) or not set(ids).issubset(train_ids)):
        raise ValueError("초기 가중치는 동일 명세의 train 소수 문장 검사 출처여야 합니다.")
    return model, {"initialization": "train_subset_weights",
                   "initial_checkpoint": str(Path(init_weights).resolve()),
                   "initial_checkpoint_sha256": sha256_file(init_weights),
                   "initial_training_data": source, "initial_training_run": source_run,
                   "optimizer_state_restored": False}


def _same_file_path(first: str | os.PathLike, second: str | os.PathLike) -> bool:
    """상대 경로·심볼릭 링크·기존 하드 링크로 같은 파일을 가리키는지 검사한다."""
    left, right = Path(first).resolve(), Path(second).resolve()
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def train(args) -> dict | None:
    """명세 검증을 통과한 자료로만 학습하며 체크포인트에 분할 출처를 저장한다."""
    train_files, val_files, provenance = training_files(args)
    if getattr(args, "check_data", False):
        print(f"[검증 완료] train={len(train_files)}, val={len(val_files)}, test는 학습에서 제외")
        print(f"[명세] {provenance.get('manifest_sha256', 'smoke_only')}")
        return
    bounded = bool(args.max_steps)
    last_out = getattr(args, "last_out", None)
    if last_out and os.path.abspath(last_out) in (os.path.abspath(args.out), os.path.abspath(MODEL_WEIGHTS_PATH)):
        raise ValueError("--last-out은 best/배포 가중치와 다른 경로여야 합니다.")
    if (args.smoke_data or bounded) and os.path.abspath(args.out) == os.path.abspath(MODEL_WEIGHTS_PATH):
        raise ValueError("스모크 학습은 --out에 별도 임시 가중치 경로를 지정하세요.")
    if bounded:
        provenance = {**provenance, "source_mode": provenance["mode"], "mode": "smoke_only",
                      "generalization_evaluation": False}
    if (min(args.epochs, args.batch_size, args.eval_every, args.accum_steps) < 1
            or args.max_steps < 0 or args.lr <= 0 or args.grad_clip < 0):
        raise ValueError("epochs/batch/eval/accum/lr은 양수, max-steps/grad-clip은 0 이상이어야 합니다.")
    snapshot_epochs = sorted(set(getattr(args, "snapshot_epochs", None) or []))
    if snapshot_epochs and (bounded or any(e < 1 or e > args.epochs for e in snapshot_epochs)):
        raise ValueError("중간 저장 epoch는 본 학습의 1~epochs 범위여야 합니다.")
    snapshots = {e: Path(args.out).resolve().parent / f"epoch_{e:04d}.pt" for e in snapshot_epochs}
    init_weights = getattr(args, "init_weights", None)
    if init_weights:
        outputs = {"--out": args.out, "--last-out": last_out,
                   "--report": getattr(args, "report", None),
                   **{f"epoch {epoch} 중간 저장": path for epoch, path in snapshots.items()}}
        for option, output in outputs.items():
            if output and _same_file_path(init_weights, output):
                raise ValueError(f"초기 가중치 원본을 덮어쓸 수 없습니다: --init-weights와 {option} 경로가 겹칩니다.")
    reserved = {Path(args.out).resolve(), Path(MODEL_WEIGHTS_PATH).resolve()}
    if last_out:
        reserved.add(Path(last_out).resolve())
    if any(p in reserved or p.exists() for p in snapshots.values()):
        raise FileExistsError("중간 저장 경로가 기존 파일 또는 best/last/배포 경로와 겹칩니다.")
    device, use_amp = resolve_device_precision(args.device, args.precision)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[데이터] train {len(train_files)} / val {len(val_files)} "
          f"· device={device} · batch={args.batch_size} · AMP={use_amp} · 누적={args.accum_steps}")

    # 길이 버킷 배치 샘플러: 비슷한 길이끼리 묶어 패딩·메모리 낭비 최소화
    train_lengths = [clip_length(f) for f in train_files]
    train_ds = ClipDataset(train_files)
    sampler = LengthBucketBatchSampler(train_lengths, args.batch_size, shuffle=True)
    if train_lengths:
        print(f"[길이] train T 범위 {min(train_lengths)}~{max(train_lengths)} "
              f"(평균 {sum(train_lengths) // len(train_lengths)})")

    architecture = getattr(args, "architecture", ARCHITECTURE_VERSION)
    model, initialization = initialize_model(architecture, getattr(args, "init_weights", None),
                                             provenance, train_files)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024)
    report = {
        "architecture_version": architecture, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "amp_fp16_ctc_fp32" if use_amp else "fp32", "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "learning_rate": args.lr, "max_steps": args.max_steps, "epochs_requested": args.epochs,
        "bounded_smoke": bounded, "training_data": provenance, "steps": [],
        "optimizer_steps": 0, "skipped_optimizer_steps": 0,
        "status": "running", "epochs": [], "best_epoch": None,
        "checkpoint": os.path.abspath(args.out),
        **initialization, "selection_metric": "val_mean_sentence_jamo_cer_beam10",
        "snapshot_epochs": snapshot_epochs, "snapshots": [],
    }

    best_cer = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    opt.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        sample_count = 0
        epoch_started = time.perf_counter()
        step_limit = min(len(sampler), args.max_steps - len(report["steps"])) if bounded else len(sampler)
        # 마지막 짧은 누적 그룹도 실제 표본 수로 가중하여 기울기 크기를 보존한다.
        # 순서를 섞은 배치 목록을 먼저 확정한다(대부분 동일 크기, 마지막 배치만 작을 수 있음).
        batches = list(iter(sampler))
        for index, indices in enumerate(batches[:step_limit]):
            batch = collate([train_ds[i] for i in indices])
            group_start = index // args.accum_steps * args.accum_steps
            group_end = min(group_start + args.accum_steps, step_limit)
            group_samples = sum(len(b) for b in batches[group_start:group_end])
            step_started = time.perf_counter()
            loss = ctc_training_loss(model, batch, device, use_amp)
            scaler.scale(loss * len(indices) / group_samples).backward()
            grad_norm = None
            if index + 1 == group_end:
                scaler.unscale_(opt)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip or float("inf"), error_if_nonfinite=False,
                )
                old_scale = scaler.get_scale()
                if not use_amp and not torch.isfinite(grad_norm):
                    raise FloatingPointError("학습 기울기에 NaN/Inf가 발생했습니다.")
                scaler.step(opt)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                report["skipped_optimizer_steps"] += int(skipped)
                report["optimizer_steps"] += int(not skipped)
                opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            report["steps"].append({
                "epoch": epoch, "loss": loss.item(), "frames": batch[2].tolist(),
                "seconds": time.perf_counter() - step_started,
                "grad_norm": float(grad_norm) if grad_norm is not None and torch.isfinite(grad_norm) else None,
            })
            total += loss.item() * len(indices)
            sample_count += len(indices)
            if bounded:
                print(f"  step {len(report['steps'])}/{args.max_steps} loss={loss.item():.4f}", flush=True)
        avg = total / sample_count

        if bounded:
            if len(report["steps"]) >= args.max_steps:
                break
            continue  # 연결 검증 실행에서는 val/test 평가 및 best 모델 선택을 하지 않는다.

        epoch_record = {"epoch": epoch, "train_loss": avg,
                        "training_seconds": time.perf_counter() - epoch_started}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            if val_files:
                val_cer, samples = evaluate_val(model, val_files, device)
                marker = ""
                if val_cer < best_cer:
                    best_cer = val_cer
                    report["best_epoch"] = epoch
                    model.save_checkpoint(args.out, training_data=provenance, training_run={
                        "bounded_smoke": False, "epoch": epoch, "val_cer": val_cer,
                        "seed": args.seed, "learning_rate": args.lr, "batch_size": args.batch_size,
                        "precision": report["precision"], **initialization,
                        "selection_metric": report["selection_metric"],
                        "optimizer_steps": report["optimizer_steps"],
                    })
                    marker = "  ★best 저장"
                epoch_record.update({"val_cer": val_cer, "samples": samples})
                print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  val_CER={val_cer:.3f}{marker}", flush=True)
            else:
                model.save_checkpoint(args.out, training_data=provenance,
                                      training_run={"bounded_smoke": False, "epoch": epoch})
                print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}")
        epoch_record["optimizer_steps"] = report["optimizer_steps"]
        if epoch in snapshots:
            model.save_checkpoint(str(snapshots[epoch]), training_data=provenance, training_run={
                "bounded_smoke": False, "epoch": epoch, "selection": "fixed_epoch",
                "optimizer_steps": report["optimizer_steps"], "seed": args.seed,
                "learning_rate": args.lr, "batch_size": args.batch_size,
                "precision": report["precision"], **initialization,
            })
            report["snapshots"].append({"epoch": epoch, "optimizer_steps": report["optimizer_steps"],
                                        "checkpoint": str(snapshots[epoch])})
        report["epochs"].append(epoch_record)
        report["epochs_completed"] = epoch
        report["best_val_cer"] = best_cer if val_files else None
        report["elapsed_seconds"] = time.perf_counter() - started
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if device.type == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    report["elapsed_seconds"] = time.perf_counter() - started
    report["best_val_cer"] = best_cer if val_files and not bounded else None
    report["checkpoint"] = os.path.abspath(args.out)
    report["status"] = "completed"
    if last_out:
        Path(last_out).parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(last_out, training_data=provenance, training_run={
            "bounded_smoke": bounded, "epoch": epoch, "selection": "last_epoch",
            "seed": args.seed, "learning_rate": args.lr, "batch_size": args.batch_size,
            "precision": report["precision"], **initialization,
            "optimizer_steps": report["optimizer_steps"],
        })
        report["last_checkpoint"] = os.path.abspath(last_out)
    if bounded:
        if report["optimizer_steps"] == 0:
            raise RuntimeError("실제 가중치 갱신이 한 번도 성공하지 않았습니다.")
        model.save_checkpoint(args.out, training_data=provenance, training_run=report)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if bounded:
        print(f"[GPU 연결 검증 완료] {report['optimizer_steps']}회 갱신 · {args.out}")
        return report
    # val이 없으면 마지막 모델이 저장됨. 샘플 디코딩 출력.
    model = load_model_checkpoint(args.out).to(device)
    eval_files = val_files or train_files[:3]
    _cer_final, samples = evaluate_val(model, eval_files[:3], device)
    print(f"[저장] best 체크포인트 → {args.out}  (best val_CER={best_cer if val_files else 'N/A'})")
    for ref, pred in samples:
        print(f"  정답='{ref}'")
        print(f"  예측='{pred}'")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LipNet CTC 학습")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--manifest", default=DEFAULT_MANIFEST, help="화자별 train/val/test 고정 명세")
    source.add_argument("--smoke-data", help="분할 없는 디버깅/과적합 확인 전용 npz 폴더")
    p.add_argument("--check-data", action="store_true", help="캐시/분할 검사만 하고 모델을 만들지 않음")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--architecture", choices=ARCHITECTURES, default=ARCHITECTURE_VERSION)
    p.add_argument("--init-weights", help="동일 명세 train 소수 문장 검사 가중치로 초기화 (optimizer 재개 아님)")
    p.add_argument("--out", default=MODEL_WEIGHTS_PATH)
    p.add_argument("--last-out", help="마지막 epoch 가중치도 별도 경로에 보관")
    p.add_argument("--snapshot-epochs", nargs="+", type=int,
                   help="비교할 epoch의 추론용 가중치를 epoch_NNNN.pt로 별도 저장 (학습 재개 파일 아님)")
    p.add_argument("--device", default=None, help="cuda | cpu (기본: 자동)")
    p.add_argument("--precision", choices=["auto", "fp32", "amp"], default="auto")
    p.add_argument("--accum-steps", type=int, default=1, help="기울기 누적 횟수")
    p.add_argument("--max-steps", type=int, default=0, help="미니배치 수 상한. 연결 검증 전용, 별도 --out 필수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", help="실행 환경/손실/메모리 기록 JSON")
    p.add_argument("--eval-every", type=int, default=10, help="검증 주기(epoch)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="길이 상한(0=무제한). 명세 사용 시 초과가 있으면 오류; smoke만 필터링")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="gradient clipping max-norm(0=비활성). CTC 학습 안정화")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train(parse_args())


if __name__ == "__main__":
    main()
