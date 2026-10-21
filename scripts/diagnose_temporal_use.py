"""11단계: 고정 가중치의 시간 정보 대조와 CTC/특징 진단. 학습/test 추론 없음."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from config import NORMALIZE_MEAN, NORMALIZE_STD
from models.cache import load_clip
from models.dataset_manifest import load_manifest, sha256_file, TEMPORAL_METHOD
from models.decoder import ctc_beam_search_decode, ctc_greedy_decode
from models.registry import load_model_checkpoint
from models.tokenizer import cer, wer, encode_text, ctc_min_input_len
from test_model import aggregate_predictions, validate_evaluation_provenance
from temporal_diagnostics import transform_frames, analyze_ctc, compare_posteriors

MANIFEST = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
FOLDERS = {
    "compact": ROOT / "results/step9-diagnostics/compact-warm-v4",
    "residual": ROOT / "results/step10-residual/run01",
}
CONDITIONS = [("normal", 42), ("static_middle", 42), ("shuffle", 42),
              ("shuffle", 137), ("shuffle", 2026), ("reversed", 42), ("constant_gray", 42)]
SOURCE_PATHS = ["scripts/diagnose_temporal_use.py", "scripts/temporal_diagnostics.py",
                "scripts/test_model.py", "backend/models/baseline.py", "backend/models/residual.py",
                "backend/models/registry.py", "backend/models/cache.py", "backend/models/decoder.py",
                "backend/models/tokenizer.py", "backend/models/dataset_manifest.py", "backend/config.py"]


def require(condition: bool, message: str) -> None:
    """최적화 실행에서도 출처와 진단 조건 검사를 생략하지 않는다."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    """UTF-8 JSON 객체를 읽는다."""
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON 객체 필요: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    """완료 결과는 덮어쓰지 않으며 NaN도 허용하지 않는다."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def length_bin(frames: int) -> str:
    """기존 10단계와 같은 50프레임 길이 구간."""
    require(1 <= frames <= 200, "1~200 프레임 명세가 필요합니다.")
    start = ((frames - 1) // 50) * 50 + 1
    return f"{start:03d}-{start + 49:03d}"


def select_cohorts(records: list[dict]) -> dict:
    """성능과 무관하게 화자×길이별 최대2개와 사전학습 short16을 고정한다."""
    train = [r for r in records if r["split"] == "train"]
    groups = defaultdict(list)
    for row in train:
        groups[(row["speaker_id"], length_bin(row["frames"]))].append(row)
    selected = []
    strata = []
    for (speaker, bucket), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: hashlib.sha256(
            f"step11-cohort-42:{r['id']}".encode("utf-8")).hexdigest())
        chosen = [r["id"] for r in ordered[:2]]
        selected.extend(chosen)
        strata.append({"speaker": speaker, "length_bin": bucket, "available": len(rows), "ids": chosen})
    short = [r["id"] for r in sorted(train, key=lambda r: (r["frames"], r["id"]))[:16]]
    return {"train_stratified": sorted(selected), "short16": short,
            "train_controls_union": sorted(set(selected) | set(short)), "strata": strata,
            "val": [r["id"] for r in records if r["split"] == "val"]}


def source_hashes() -> dict:
    """진단과 모델 계산에 직접 사용하는 코드의 해시."""
    return {name: sha256_file(ROOT / name) for name in SOURCE_PATHS}


def condition_name(condition: str, seed: int) -> str:
    """각 shuffle 반복을 별도 조건으로 기록한다."""
    return f"shuffle_{seed}" if condition == "shuffle" else condition


def verify_cache(row: dict) -> dict:
    """train/val만 열고 명세와 캐시의 픽셀 출처/정답을 대조한다."""
    require(row["split"] in ("train", "val"), "test 캐시 접근 금지")
    path = MANIFEST.parent / row["cache_path"]
    require(sha256_file(path) == row["cache_sha256"], f"캐시 해시 불일치: {row['id']}")
    item = load_clip(path)
    require(item["text"] == row["text"] and item["source_text"] == row["source_text"]
            and len(item["frames"]) == row["frames"], "캐시 내용 불일치")
    with np.load(path, allow_pickle=False) as data:
        for key in ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition",
                    "roi_method", "label_sha256", "temporal_method", "video_fps", "label_fps"):
            require(str(data[key]) == str(row[key]), f"캐시 메타데이터 불일치: {row['id']}:{key}")
    return item


def feature_stats(values: np.ndarray) -> dict:
    """각 단계 내 RMS와 시간 변화량. 단계 간 정보량의 직접 비교가 아니다."""
    x = values.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    centered = float(np.sqrt(np.mean((x - x.mean(axis=0, keepdims=True)) ** 2)))
    adjacent = float(np.sqrt(np.mean(np.diff(x, axis=0) ** 2))) if len(x) > 1 else 0.0
    return {"shape": list(x.shape), "rms": rms, "temporal_centered_rms": centered,
            "temporal_centered_relative_rms": centered / max(rms, 1e-12),
            "adjacent_rms": adjacent, "adjacent_relative_rms": adjacent / max(rms, 1e-12)}


def feature_distance(reference: np.ndarray, changed: np.ndarray) -> dict:
    """같은 모델/단계의 두 입력 사이 특징 차이를 계산한다."""
    require(reference.shape == changed.shape, "특징 형상 불일치")
    a, b = reference.astype(np.float64), changed.astype(np.float64)
    delta = float(np.sqrt(np.mean((a - b) ** 2)))
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)
    return {"rms_difference": delta,
            "relative_rms_difference": delta / max(float(np.sqrt(np.mean(a * a))), 1e-12),
            "mean_frame_cosine": float(cosine.mean())}


@torch.inference_mode()
def forward(model, frames: np.ndarray, capture: bool) -> tuple[np.ndarray, dict]:
    """기존 평가처럼 패딩 없는 B=1 FP32 추론; hook은 값을 수정하지 않는다."""
    features, handles = {}, []

    def save(name: str, value: torch.Tensor, spatial: bool = False) -> None:
        require(name not in features, f"중복 hook: {name}")
        value = value.detach().float().cpu()
        if spatial:
            value = value.permute(0, 2, 1, 3, 4).flatten(2)
        features[name] = value[0].numpy().copy()

    if capture:
        handles = [model.spatial_pool.register_forward_hook(lambda m, a, o: save("cnn", o, True)),
                   model.projection.register_forward_hook(lambda m, a, o: save("projection", o)),
                   model.fc.register_forward_pre_hook(lambda m, a: save("lstm", a[0])),
                   model.fc.register_forward_hook(lambda m, a, o: save("logits", o))]
    try:
        values = (frames.astype(np.float32) / 255.0 - np.asarray(NORMALIZE_MEAN, dtype=np.float32))
        values /= np.asarray(NORMALIZE_STD, dtype=np.float32)
        tensor = torch.from_numpy(values).permute(3, 0, 1, 2).unsqueeze(0).to("cuda")
        log_probs = model(tensor).squeeze(1).float().cpu().numpy()
        require(np.isfinite(log_probs).all() and log_probs.shape == (len(frames), 69), "비정상 출력")
        if capture:
            require(set(features) == {"cnn", "projection", "lstm", "logits"}, "특징 hook 누락")
        return log_probs, features
    finally:
        for handle in handles:
            handle.remove()


def prediction_row(source: dict, prediction: str) -> dict:
    """기존 보고서와 동일한 CER/WER 집계용 행."""
    reference = source["text"]
    characters = len(unicodedata.normalize("NFD", reference))
    words = len(reference.split())
    c, w = cer(reference, prediction), wer(reference, prediction)
    return {"id": source["id"], "reference": reference, "prediction": prediction,
            "frames": source["frames"], "cer": c, "wer": w, "exact": reference == prediction,
            "reference_jamo_count": characters, "reference_word_count": words,
            "jamo_edit_distance": round(c * characters), "word_edit_distance": round(w * words)}


def aggregate_rows(rows: list[dict]) -> dict:
    """빈 집합을 숨기지 않고 집단별 예측 수와 편집 지표를 계산한다."""
    return aggregate_predictions(rows) if rows else {"clips": 0}


def run_model(key: str, plan: dict, records: list[dict], summary: dict, out: Path,
              *, short_only: bool = False) -> dict:
    """고정 모델 한 개를 평가하고 raw log posterior와 문장별 진단을 보존한다."""
    spec = plan["models"][key]
    checkpoint = Path(spec["checkpoint"])
    require(sha256_file(checkpoint) == spec["checkpoint_sha256"], "가중치 변경")
    model = load_model_checkpoint(str(checkpoint)).to("cuda").eval()
    if short_only:
        provenance = model.training_data
        require(provenance.get("mode") == "smoke_only" and provenance.get("purpose") == "train_subset_overfit"
                and provenance.get("manifest_sha256") == summary["manifest_sha256"]
                and provenance.get("sample_ids") == plan["cohorts"]["short16"], "short16 출처 불일치")
        selected = {r["id"]: r for r in records if r["split"] == "train"}
        todo = [selected[s] for s in plan["cohorts"]["short16"]]
        baselines = {}
    else:
        validate_evaluation_provenance(model.training_data, summary)
        require(model.training_run.get("bounded_smoke") is False, "본 학습 가중치 필요")
        todo = [r for r in records if r["split"] in ("val", "train")]
        baselines = {}
        for split in ("train", "val"):
            reference = read_json(Path(spec["folder"]) / f"best_{split}.json")
            require(reference["checkpoint_sha256"] == spec["checkpoint_sha256"]
                    and reference["manifest_sha256"] == summary["manifest_sha256"]
                    and reference["control"] is None and reference["decoder"] == "beam"
                    and reference["beam_width"] == 10, "기존 정상 평가 출처 불일치")
            expected = [r for r in records if r["split"] == split]
            require([r["id"] for r in reference["predictions"]] == [r["id"] for r in expected], "기존 평가 ID 불일치")
            for saved, original in zip(reference["predictions"], expected):
                check = prediction_row(original, saved["prediction"])
                require(all(saved[name] == value for name, value in check.items()), "기존 평가 정답/지표 불일치")
                baselines[saved["id"]] = saved["prediction"]
            require(aggregate_predictions(reference["predictions"]) == reference["metrics"], "기존 평가 집계 불일치")
    rows, arrays, normal_matches = [], {}, 0
    controls = set(plan["cohorts"]["train_controls_union"])
    started = time.perf_counter()
    folder = out / key
    folder.mkdir()
    with (folder / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for index, source in enumerate(todo):
            item = verify_cache(source)
            do_controls = short_only or source["split"] == "val" or source["id"] in controls
            normal_probs, normal_features, normal_text = None, None, None
            for condition, seed in (CONDITIONS if do_controls else CONDITIONS[:1]):
                name = condition_name(condition, seed)
                frames, transform = transform_frames(item["frames"], condition, sample_id=source["id"], seed=seed)
                probs, features = forward(model, frames, capture=do_controls)
                prediction = ctc_beam_search_decode(probs, beam_width=10)
                row = prediction_row(source, prediction)
                greedy = ctc_greedy_decode(probs)
                row.update({"split": source["split"], "speaker_id": source["speaker_id"],
                            "length_bin": length_bin(source["frames"]), "condition": name,
                            "short16": source["id"] in plan["cohorts"]["short16"],
                            "stratified": source["id"] in plan["cohorts"]["train_stratified"],
                            "transform": transform, "ctc": analyze_ctc(probs, item["label"], beam_text=prediction),
                            "greedy_prediction": greedy, "greedy_cer": cer(source["text"], greedy),
                            "greedy_exact": greedy == source["text"], "greedy_beam_same": greedy == prediction,
                            "features": {stage: feature_stats(v) for stage, v in features.items()}})
                target = torch.tensor(item["label"], dtype=torch.long)
                loss = torch.nn.functional.ctc_loss(torch.from_numpy(probs[:, None, :]), target,
                    torch.tensor([len(probs)]), torch.tensor([len(target)]), blank=0, reduction="none", zero_infinity=False)
                row["ctc_loss_per_target_token"] = float(loss.item() / len(target))
                if condition == "normal":
                    normal_probs, normal_features, normal_text = probs, features, prediction
                    if not short_only:
                        require(prediction == baselines[source["id"]], f"정상 예측 재현 실패: {key}/{source['id']}")
                        normal_matches += 1
                else:
                    row["comparison_to_normal"] = compare_posteriors(normal_probs, probs,
                        normal_text=normal_text, control_text=prediction)
                    row["feature_comparison_same_position"] = {
                        stage: feature_distance(normal_features[stage], value) for stage, value in features.items()}
                    if condition in ("shuffle", "reversed"):
                        permutation = (np.asarray(transform["permutation"], dtype=int) if condition == "shuffle"
                                       else np.arange(len(probs) - 1, -1, -1))
                        row["feature_comparison_source_frame_correspondence"] = {
                            stage: feature_distance(normal_features[stage][permutation], value)
                            for stage, value in features.items()}
                arrays[f"{source['id']}__{name}"] = probs
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 50 == 0 or index + 1 == len(todo):
                print(json.dumps({"model": key, "clips": index + 1, "total": len(todo),
                                  "forwards": len(rows), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    groups = {}
    cohort_names = ("short16",) if short_only else ("val", "train_all_normal", "train_stratified", "short16")
    for cohort in cohort_names:
        for condition, seed in CONDITIONS:
            name = condition_name(condition, seed)
            chosen = [r for r in rows if r["condition"] == name and (
                (cohort == "val" and r["split"] == "val")
                or (cohort == "train_all_normal" and r["split"] == "train" and name == "normal")
                or (cohort == "train_stratified" and r["stratified"])
                or (cohort == "short16" and r["short16"]))]
            if chosen:
                groups[f"{cohort}/{name}"] = aggregate_rows(chosen)
    report = {"model": key, "checkpoint": spec, "training_data": model.training_data,
              "training_run": model.training_run, "manifest_sha256": summary["manifest_sha256"],
              "status": "completed", "device": "cuda", "precision": "fp32", "batch_size": 1,
              "beam_width": 10, "training_performed": False, "test_inference_performed": False,
              "short_overfit_diagnostic_only": short_only, "normal_prediction_reproduction_matches": normal_matches,
              "clips": len(todo), "forwards": len(rows), "groups": groups,
              "elapsed_seconds": time.perf_counter() - started,
              "rows_sha256": sha256_file(folder / "rows.jsonl"),
              "posteriors_sha256": sha256_file(folder / "posteriors.npz")}
    require(sha256_file(checkpoint) == spec["checkpoint_sha256"], "평가 후 가중치 변경")
    write_json(folder / "report.json", report)
    del model, arrays
    torch.cuda.empty_cache()
    return report


def main() -> None:
    """새 결과 폴더에 계획을 먼저 확정한 뒤 train/val 추론만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    require(torch.cuda.is_available(), "검증된 CUDA 환경이 필요합니다.")
    out = Path(args.out).resolve()
    require(out.is_relative_to(ROOT / "results") and not out.exists(), "새 results 하위 폴더 필요")
    document, summary = load_manifest(MANIFEST, verify_files=False)
    require(document["policy"]["temporal_method"] == TEMPORAL_METHOD, "v4 시간 교정 명세 필요")
    cohorts = select_cohorts(document["records"])
    specs = {}
    for key, folder in FOLDERS.items():
        checkpoint = folder / "best.pt"
        training = read_json(folder / "training.json")
        require(training.get("status") == "completed" and training.get("epochs_completed") == 20, "완료된20epoch 필요")
        require(training["initial_training_data"]["sample_ids"] == cohorts["short16"], "선행16 선정 불일치")
        validate_evaluation_provenance(training["training_data"], summary)
        specs[key] = {"folder": str(folder), "checkpoint": str(checkpoint),
                      "checkpoint_sha256": sha256_file(checkpoint), "training_report_sha256": sha256_file(folder / "training.json"),
                      "reference_reports": {split: sha256_file(folder / f"best_{split}.json") for split in ("train", "val")}}
        short_path = Path(training["initial_checkpoint"])
        require(sha256_file(short_path) == training["initial_checkpoint_sha256"], "선행 가중치 변경")
        specs[f"{key}_short16"] = {"checkpoint": str(short_path), "checkpoint_sha256": sha256_file(short_path)}
    plan = {"stage": 11, "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "fixed_checkpoint_temporal_and_ctc_diagnosis", "manifest": str(MANIFEST),
            "manifest_sha256": summary["manifest_sha256"], "cohorts": cohorts, "models": specs,
            "conditions": [{"name": condition_name(c, s), "transform": c, "seed": s} for c, s in CONDITIONS],
            "selection": "train speaker x 50-frame bin: SHA256 order first2; union fixed shortest16; all val20",
            "training_performed": False, "test_inference_performed": False, "model_selection_performed": False,
            "sources": source_hashes(), "environment": {"torch": torch.__version__, "numpy": np.__version__,
                "python": sys.version, "gpu": torch.cuda.get_device_name(0), "threads": torch.get_num_threads()},
            "limits": ["controls are distribution shifts, not proof of phonetic decoding",
                       "shuffle source-frame correspondence still changes temporal context",
                       "feature differences are not information-theoretic loss or gradient evidence",
                       "train stratified and short16 cohorts may overlap and must not be added",
                       "development val20 has been used in checkpoint selection; not independent test"]}
    out.mkdir(parents=True)
    write_json(out / "plan.json", plan)
    print(json.dumps({"out": str(out), "stratified": len(cohorts["train_stratified"]),
                      "train_controls_union": len(cohorts["train_controls_union"]), "val": len(cohorts["val"])}), flush=True)
    reports = {}
    for key in ("compact", "residual", "compact_short16", "residual_short16"):
        reports[key] = run_model(key, plan, document["records"], summary, out, short_only=key.endswith("short16"))
    require(source_hashes() == plan["sources"] and sha256_file(MANIFEST) == plan["manifest_sha256"], "평가 중 소스/명세 변경")
    write_json(out / "completed.json", {"status": "completed", "plan_sha256": sha256_file(out / "plan.json"),
               "models": reports, "test_inference_performed": False, "training_performed": False})
    print("11단계 고정 가중치 진단 완료", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[진단 실패] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
