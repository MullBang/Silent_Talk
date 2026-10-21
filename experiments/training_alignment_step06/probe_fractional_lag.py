"""Post-hoc diagnostic of two already-saved first-anchor PCM pairs only.

Does not change the frozen .99 gate, any original source, audit result or
quarantine. Reconstructing a finer delay grid from 8kHz PCM is a diagnostic,
not new source evidence and not a replacement for a prospective audit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

BASE = Path(__file__).resolve().parent
SOURCES = ("lip_J_1_F_02_C002_A_004", "lip_J_1_F_02_C004_A_001")
RATE, UP, RADIUS = 8000, 6, .002


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pcm(path):
    path = Path(path).resolve()
    if not path.is_relative_to((BASE / "results/pcm").resolve()):
        raise ValueError("Only this audit's existing local PCM may be read")
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("Unexpected PCM schema")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float64) / 32768


def local_pearson(reference, search, first, last):
    """Pearson at every inclusive local integer shift, using FFT numerator."""
    n = len(reference)
    if not 0 <= first <= last <= len(search) - n:
        raise ValueError("Local search bounds invalid")
    piece = search[first:last+n]
    centered = reference - np.mean(reference)
    sums = np.r_[0., np.cumsum(piece)]
    squares = np.r_[0., np.cumsum(piece * piece)]
    energies = np.maximum(squares[n:] - squares[:-n] - (sums[n:] - sums[:-n])**2 / n, 0)
    numerator = correlate(piece, centered, mode="valid", method="fft")
    scores = numerator / np.sqrt(np.dot(centered, centered) * energies)
    winner = int(np.argmax(np.abs(scores)))
    return {"best_start_sample": first + winner, "signed_correlation": float(scores[winner]),
            "absolute_correlation": float(abs(scores[winner])), "candidate_count": len(scores),
            "best_at_search_boundary": winner in (0, len(scores) - 1)}


def refine(reference, search, coarse_start):
    filt = butter(4, [100, 3400], btype="bandpass", fs=RATE, output="sos")
    ref = sosfiltfilt(filt, reference)
    sea = sosfiltfilt(filt, search)
    coarse = local_pearson(ref, sea, coarse_start, coarse_start)
    ref_up = resample_poly(ref, UP, 1)
    sea_up = resample_poly(sea, UP, 1)
    radius = round(RADIUS * RATE * UP)
    fine = local_pearson(ref_up, sea_up, coarse_start*UP-radius, coarse_start*UP+radius)
    coarse["sample_rate"] = RATE
    coarse["start_seconds"] = coarse_start / RATE
    fine["sample_rate"] = RATE * UP
    fine["start_seconds"] = fine["best_start_sample"] / (RATE * UP)
    fine["start_change_seconds"] = fine["start_seconds"] - coarse["start_seconds"]
    fine["absolute_correlation_gain"] = fine["absolute_correlation"] - coarse["absolute_correlation"]
    # A direct dot product independently verifies the winning FFT Pearson.
    matched = sea_up[fine["best_start_sample"]:fine["best_start_sample"]+len(ref_up)]
    direct = float(np.corrcoef(ref_up, matched)[0, 1])
    if abs(direct-fine["signed_correlation"]) > 1e-10:
        raise ValueError("FFT/direct Pearson mismatch")
    fine["direct_signed_correlation"] = direct
    return {"coarse": coarse, "fine": fine}


def synthetic_check():
    """Sample one known analytic band-limited waveform on offset 8kHz grids."""
    rng = np.random.default_rng(6006)
    frequencies = rng.uniform(150, 3200, 73)
    phases = rng.uniform(-np.pi, np.pi, len(frequencies))
    amplitudes = rng.uniform(.3, 1, len(frequencies))
    known_start = 2 + 1024/48000
    def signal(times):
        out = np.zeros_like(times)
        for f, phase, amplitude in zip(frequencies, phases, amplitudes):
            out += amplitude * np.sin(2*np.pi*f*times + phase)
        return out / len(frequencies)
    reference = signal(np.arange(6*RATE)/RATE + known_start)
    search = signal(np.arange(10*RATE)/RATE)
    result = refine(reference, search, round(known_start*RATE))
    error = result["fine"]["start_seconds"] - known_start
    if abs(error) > 1/(RATE*UP) or result["fine"]["absolute_correlation"] < .995:
        raise ValueError("Known fractional-delay validation failed")
    return {"seed": 6006, "known_start_seconds": known_start, "construction":
            "73 deterministic sinusoids in 150-3200Hz sampled directly at 8kHz with a known fractional offset; no VSR or source data",
            "fine_start_error_seconds": error, "passed": True, **result}


def main():
    evidence = [BASE/"plan.json", BASE/"audit_sources.py"]
    evidence += [BASE/f"results/sources/{stem}.json" for stem in SOURCES]
    before = {str(path): sha(path) for path in evidence}
    rows = []
    for stem in SOURCES:
        path = BASE/f"results/sources/{stem}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        anchor = saved["anchors"][0]
        reference, search = anchor["reference"], anchor["mp4_search"]
        if reference["start_seconds"] != 30 or search["start_seconds"] != 15:
            raise ValueError("Only the first frozen anchor is allowed")
        for item in (reference, search):
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Existing local PCM hash changed")
        score = anchor["score"]
        probe = refine(read_pcm(reference["path"]), read_pcm(search["path"]), score["best_start_sample_in_search"])
        if abs(probe["coarse"]["absolute_correlation"] - score["absolute_correlation"]) > 1e-9:
            raise ValueError("Original coarse Pearson did not reproduce")
        for grid in ("coarse", "fine"):
            probe[grid]["offset_mp4_minus_original_seconds"] = search["start_seconds"] + probe[grid]["start_seconds"] - reference["start_seconds"]
        rows.append({"source_stem": stem, "anchor_seconds": 30, "original_status_preserved": saved["status"],
                     "original_anchor_strict_pass_preserved": score["strict_anchor_pass"],
                     "evidence": [{"path": item["path"], "sha256": item["sha256"]} for item in (reference, search)], **probe})
    synthetic = synthetic_check()
    if before != {str(path): sha(path) for path in evidence}:
        raise ValueError("Frozen original evidence changed")
    result = {"schema": "lipnet-step06-posthoc-fractional-lag-probe-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": {"posthoc_diagnostic_only": True, "sources": 2, "anchors_per_source": 1,
                        "new_original_media_reads": False, "threshold_changes": False, "quarantine_changes": False},
              "method": {"filter": "Same 4th-order Butterworth100-3400Hz zero-phase at8kHz on both full excerpts",
                         "interpolation": "scipy.signal.resample_poly(x, 6, 1), default Kaiser5.0, independently for both signals after filtering",
                         "grid_hz": 48000, "local_radius_seconds": RADIUS, "score": "absolute centered Pearson at193 shifts; direct np.corrcoef verification of winning shift",
                         "frozen_minimum_correlation_unchanged": .99},
              "unchanged_evidence": [{"path": path, "sha256": checksum} for path, checksum in before.items()],
              "sources": rows, "synthetic_validation": synthetic,
              "interpretation_limits": ["Two first anchors only; no whole-recording or full-cohort validation.",
                 "Upsampling reconstructs an interpolant of existing8kHz audio, not new48kHz source observations.",
                 "Higher refined correlation can support a sample-grid explanation, but does not certify transcript correctness or lip synchronization.",
                 "Original frozen-gate decisions and all quarantines remain unchanged; any replacement estimator needs a new prospective plan."]}
    output = BASE/"results/fractional_probe.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "sources": rows,
                      "synthetic_validation": synthetic}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
