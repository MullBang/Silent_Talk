"""Read-only audit of local AIHub TAR labels and their media metadata.

Does not assert transcript truth, lip/audio synchronization or decoded payload
integrity. The latter is audited independently. Outputs are local audit evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import time

import numpy as np

IDENTITY = re.compile(r"^(lip_[A-Z]+_\d+_[FM]_\d+_([A-Z]\d+))_([A-Z])_(\d+)$")
DEFAULT_ROOT = Path(r"D:\009.립리딩(입모양) 음성인식 데이터")
DEFAULT_OUT = Path(__file__).parent / "label_results"


def identity(stem):
    m = IDENTITY.fullmatch(stem)
    if not m:
        return None
    prefix, speaker, angle, number = m.groups()
    return {"speaker": speaker, "angle": angle,
            "recording": f"{prefix}_{number}", "frontal_stem": f"{prefix}_A_{number}"}


def boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    while start < end:
        if end - start < 8:
            raise ValueError("incomplete atom header")
        size, name = struct.unpack_from(">I4s", data, start)
        head = 8
        if size == 1:
            if end - start < 16:
                raise ValueError("incomplete extended atom header")
            size = struct.unpack_from(">Q", data, start + 8)[0]
            head = 16
        elif size == 0:
            size = end - start
        if size < head or start + size > end:
            raise ValueError("atom outside parent")
        yield name, start + head, start + size
        start += size


def video_metadata(fileobj, size):
    """Read moov only; count encoded video samples, not decoded frames."""
    cursor = 0
    moov = None
    while cursor + 8 <= size:
        fileobj.seek(cursor)
        header = fileobj.read(16)
        length, kind = struct.unpack_from(">I4s", header)
        head = 8
        if length == 1:
            length = struct.unpack_from(">Q", header, 8)[0]
            head = 16
        elif length == 0:
            length = size - cursor
        if length < head or cursor + length > size:
            raise ValueError("top-level atom outside member")
        if kind == b"moov":
            if length - head > 16 * 1024 * 1024:
                raise ValueError("moov exceeds safe metadata read limit")
            fileobj.seek(cursor + head)
            moov = fileobj.read(length - head)
            break
        cursor += length
    if moov is None:
        raise ValueError("no moov")
    tracks = []
    for name, a, b in boxes(moov):
        if name != b"trak":
            continue
        track = {}
        def walk(lo, hi):
            for typ, p, q in boxes(moov, lo, hi):
                if typ in (b"mdia", b"minf", b"stbl"):
                    walk(p, q)
                elif typ == b"hdlr":
                    track["handler"] = moov[p + 8:p + 12].decode("ascii")
                elif typ == b"tkhd":
                    track["width"] = struct.unpack_from(">I", moov, q - 8)[0] / 65536
                    track["height"] = struct.unpack_from(">I", moov, q - 4)[0] / 65536
                elif typ == b"mdhd":
                    v = moov[p]
                    if v == 0:
                        scale, duration = struct.unpack_from(">II", moov, p + 12)
                    elif v == 1:
                        scale = struct.unpack_from(">I", moov, p + 20)[0]
                        duration = struct.unpack_from(">Q", moov, p + 24)[0]
                    else:
                        raise ValueError("unsupported mdhd version")
                    track["timescale"] = scale
                    track["duration"] = duration / scale if scale else None
                elif typ == b"stsz":
                    track["sample_count"] = struct.unpack_from(">I", moov, p + 8)[0]
                elif typ == b"stts":
                    n = struct.unpack_from(">I", moov, p + 4)[0]
                    if p + 8 + n * 8 > q:
                        raise ValueError("stts table outside atom")
                    pairs = [struct.unpack_from(">II", moov, p + 8 + i * 8) for i in range(n)]
                    track["stts_sample_count"] = sum(x for x, _ in pairs)
                    track["stts_ticks"] = sum(x * y for x, y in pairs)
                    track["sample_deltas"] = sorted({y for _, y in pairs})
        walk(a, b)
        if track.get("handler") == "vide":
            if track.get("stts_ticks") and track.get("timescale"):
                track["sample_duration"] = track["stts_ticks"] / track["timescale"]
                track["average_fps"] = track["stts_sample_count"] / track["sample_duration"]
            tracks.append(track)
    if len(tracks) != 1:
        raise ValueError(f"expected one video track, found {len(tracks)}")
    return tracks[0]


def finite_number(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def parse_duration(value):
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError("duration must be h:mm:ss")
    vals = [float(v) for v in parts]
    if not all(math.isfinite(v) and v >= 0 for v in vals):
        raise ValueError("invalid duration")
    return vals[0] * 3600 + vals[1] * 60 + vals[2]


def validate_label(data, stem, media, video_names, audio_names):
    errors, warnings, detail = [], [], {}
    if isinstance(data, list):
        if len(data) != 1:
            return {"errors": ["root_list_not_singleton"], "warnings": [], "detail": {}}
        data = data[0]
    if not isinstance(data, dict):
        return {"errors": ["root_not_object"], "warnings": [], "detail": {}}
    ident = identity(stem)
    detail["identity"] = ident
    if not ident:
        errors.append("filename_identity_invalid")
    for key in ("Video_info", "Audio_info", "Video_env", "speaker_info", "Bounding_box_info"):
        if not isinstance(data.get(key), dict):
            errors.append(f"{key}_not_object")
    if errors:
        return {"errors": errors, "warnings": warnings, "detail": detail}
    vi, ai = data["Video_info"], data["Audio_info"]
    detail["video_name"] = vi.get("video_Name")
    detail["audio_name_declared"] = ai.get("Audio_Name")
    if vi.get("video_Name") != stem + ".mp4":
        errors.append("video_name_differs_from_label_filename")
    if vi.get("video_Name") not in video_names:
        errors.append("referenced_video_missing")
    if ai.get("Audio_Name") not in audio_names:
        frontal = ident["frontal_stem"] + ".wav" if ident else None
        if frontal in audio_names and ident["angle"] != "A":
            warnings.append("declared_nonfrontal_wav_absent_frontal_wav_available")
            detail["shared_audio_candidate"] = frontal
        else:
            errors.append("referenced_audio_missing")
    fps = vi.get("FPS")
    detail["label_fps"] = fps
    if not finite_number(fps) or fps <= 0:
        errors.append("label_fps_invalid")
    if ident and data["speaker_info"].get("speaker_ID") != ident["speaker"]:
        errors.append("speaker_metadata_filename_mismatch")
    if ident and data["Video_env"].get("Angle") != ident["angle"]:
        errors.append("angle_metadata_filename_mismatch")
    try:
        dimensions = [int(v) for v in str(vi.get("Resolution")).split("*")]
        if len(dimensions) != 2 or min(dimensions) <= 0:
            raise ValueError()
        width, height = dimensions
    except (ValueError, TypeError):
        errors.append("label_resolution_invalid")
        width = height = None
    try:
        detail["label_duration_seconds"] = parse_duration(vi.get("video_Duration"))
    except (ValueError, TypeError):
        errors.append("label_duration_invalid")
    if media:
        detail["video_metadata"] = media
        if width and (width != media.get("width") or height != media.get("height")):
            errors.append("label_video_dimensions_mismatch")
        if finite_number(fps) and media.get("average_fps") and abs(fps - media["average_fps"]) > 1e-6:
            warnings.append("label_fps_differs_from_video_average_use_video_timestamps")
        if media.get("sample_count") != media.get("stts_sample_count"):
            errors.append("video_sample_tables_disagree")
        if (media.get("duration") is not None and "label_duration_seconds" in detail
                and abs(media["duration"] - detail["label_duration_seconds"]) > 1.1):
            warnings.append("rounded_label_duration_differs_from_video_over_1_1_seconds")
    else:
        errors.append("video_metadata_unavailable")
    for region in ("Lip_bounding_box", "Face_bounding_box"):
        try:
            coords = data["Bounding_box_info"][region]["xtl_ytl_xbr_ybr"]
            arr = np.asarray(coords, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 4 or not len(arr):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append(f"{region}_invalid_shape")
            continue
        n = len(arr)
        detail[region] = {"count": n}
        finite = np.isfinite(arr).all(axis=1)
        valid_geometry = finite & (arr[:, 0] < arr[:, 2]) & (arr[:, 1] < arr[:, 3])
        invalid = np.flatnonzero(~valid_geometry)
        detail[region]["invalid_geometry_count"] = len(invalid)
        detail[region]["invalid_geometry_first_indices"] = invalid[:10].tolist()
        if len(invalid):
            (errors if region.startswith("Lip") else warnings).append(f"{region}_invalid_geometry")
        if width and height:
            # This dataset's established, visually verified raw order is y,x,y,x.
            inside = finite & (arr[:, 0] >= 0) & (arr[:, 1] >= 0) & (arr[:, 2] <= height) & (arr[:, 3] <= width)
            outside = np.flatnonzero(~inside)
            detail[region]["outside_image_count"] = len(outside)
            detail[region]["outside_image_first_indices"] = outside[:10].tolist()
            if len(outside):
                (errors if region.startswith("Lip") else warnings).append(f"{region}_outside_image_yxyx")
        if media and n != media.get("sample_count"):
            errors.append(f"{region}_count_differs_from_video_samples")
    sentences = data.get("Sentence_info")
    if not isinstance(sentences, list) or not sentences:
        errors.append("sentence_list_missing_or_empty")
        sentences = []
    detail["sentence_count"] = len(sentences)
    ids, strings, previous_start, previous_end = set(), [], -1, -1
    sentence_issues = []
    signature = []
    for ordinal, sent in enumerate(sentences):
        issue = []
        if not isinstance(sent, dict):
            sentence_issues.append({"ordinal": ordinal, "issues": ["not_object"]})
            continue
        sid, txt, start, end = (sent.get(k) for k in ("ID", "sentence_text", "start_time", "end_time"))
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 1:
            issue.append("invalid_sentence_id")
        elif sid in ids:
            issue.append("duplicate_sentence_id")
        else:
            ids.add(sid)
        if not isinstance(txt, str) or not txt.strip():
            issue.append("empty_or_nonstring_transcript")
        else:
            strings.append(txt)
            if "\ufffd" in txt or "\x00" in txt:
                issue.append("replacement_or_null_character_in_transcript")
        if not finite_number(start) or not finite_number(end) or start < 0 or end <= start:
            issue.append("invalid_time_interval")
        else:
            if media and media.get("duration") is not None and end > media["duration"] + 1e-6:
                issue.append("sentence_exceeds_video_duration")
            if start < previous_start:
                warnings.append("sentence_start_not_monotonic")
            elif start < previous_end - 1e-6:
                warnings.append("sentence_intervals_overlap")
            previous_start, previous_end = start, end
        if issue:
            sentence_issues.append({"ordinal": ordinal, "ID": sid, "issues": issue})
        signature.append([sid, txt, start, end])
    if sentence_issues:
        errors.append("sentence_validation_errors")
        detail["sentence_issues"] = sentence_issues
    repeated = len(strings) - len(set(strings))
    if repeated:
        warnings.append("repeated_transcript_within_recording")
        detail["repeated_transcript_count"] = repeated
    detail["sentence_signature_sha256"] = hashlib.sha256(json.dumps(signature, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "detail": detail}


def synthetic_tests():
    assert finite_number(3) and not finite_number(True) and not finite_number(float("nan"))
    assert parse_duration("0:05:03") == 303
    assert identity("lip_J_1_F_02_C040_B_001")["frontal_stem"] == "lip_J_1_F_02_C040_A_001"
    d = {"Video_info": {"video_Name": "lip_J_1_F_02_C040_A_001.mp4", "FPS": 30,
        "Resolution": "1920*1080", "video_Duration": "0:00:01"},
        "Audio_info": {"Audio_Name": "lip_J_1_F_02_C040_A_001.wav"},
        "Video_env": {"Angle": "A"}, "speaker_info": {"speaker_ID": "C040"},
        "Bounding_box_info": {k: {"xtl_ytl_xbr_ybr": [[10, 20, 30, 40]] * 30}
            for k in ("Lip_bounding_box", "Face_bounding_box")},
        "Sentence_info": [{"ID": 1, "sentence_text": "검사", "start_time": 0, "end_time": .8}]}
    md = {"width": 1920, "height": 1080, "sample_count": 30, "stts_sample_count": 30,
          "average_fps": 30, "duration": 1}
    stem = "lip_J_1_F_02_C040_A_001"
    assert not validate_label([d], stem, md, {stem + ".mp4"}, {stem + ".wav"})["errors"]
    d["Sentence_info"][0]["end_time"] = 2
    assert "sentence_validation_errors" in validate_label([d], stem, md, {stem + ".mp4"}, {stem + ".wav"})["errors"]
    d["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"][0] = [0, 0, 1081, 5]
    assert "Lip_bounding_box_outside_image_yxyx" in validate_label([d], stem, md, {stem + ".mp4"}, {stem + ".wav"})["errors"]
    try:
        list(boxes(struct.pack(">I4s", 16, b"moov")))
    except ValueError:
        pass
    else:
        raise AssertionError("truncated atom accepted")
    print("synthetic_tests_passed", flush=True)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def audit_selected_impact(out=DEFAULT_OUT):
    """Compare new D: label bytes against the frozen training/pilot selections.

    The record-level issues remain explicit. A sentence error in an unselected
    sentence is distinguished from an error in the chosen training sentence.
    Lip-box issues are located in the selected 25 fps nominal time window only;
    this is not a new audiovisual alignment approval.
    """
    root = Path(__file__).resolve().parents[2]
    rows = {r["stem"]: r for r in json.loads((out / "label_records.json").read_text(encoding="utf-8"))}
    legacy = json.loads((root / "experiments/training_alignment_step06/training_inventory.json").read_text(encoding="utf-8"))
    pilot = json.loads((root / "results/step18-intake/run01/selection.json").read_text(encoding="utf-8"))
    selected = [("legacy_stage17_B512", s["source_stem"], r) for s in legacy["sources"] for r in s["training_records"]]
    selected += [("new_stage18_pilot120", r["id"].rsplit("_", 1)[0], r) for r in pilot["records"]]
    by_source = defaultdict(list)
    for group, stem, record in selected:
        by_source[stem].append((group, record))
    results = []
    archives = {}
    try:
        for stem, chosen in sorted(by_source.items()):
            row = rows.get(stem)
            payload = None
            if row and any(e.startswith("Lip_bounding_box_") for e in row["errors"]):
                path = row["archive"]
                tf = archives.setdefault(path, None)
                if tf is None:
                    tf = archives[path] = tarfile.open(path, "r:")
                with tf.extractfile(row["member"]) as f:
                    obj = json.load(f)
                payload = obj[0] if isinstance(obj, list) else obj
            for group, record in chosen:
                res = {"group": group, "id": record["id"], "source_stem": stem,
                       "start_time": record["start_time"], "end_time": record["end_time"]}
                if row is None:
                    res["errors"] = ["label_not_found_in_downloaded_archives"]
                    results.append(res)
                    continue
                res["label_sha256_matches_frozen_selection"] = row["sha256"] == record["label_sha256"]
                res["record_level_errors"] = row["errors"]
                res["selected_sentence_issues"] = [v for v in row["detail"].get("sentence_issues", []) if v.get("ID") == record["sentence_id"]]
                # Current preprocessing consumes Lip_bounding_box only.
                # Preserve face-box quality findings without treating them as
                # a hard failure of a lip-only selected training sample.
                errs = [e for e in row["errors"] if e != "sentence_validation_errors" and not e.startswith("Face_bounding_box_")]
                if res["selected_sentence_issues"]:
                    errs.append("selected_sentence_validation_error")
                if not res["label_sha256_matches_frozen_selection"]:
                    errs.append("frozen_label_bytes_differ")
                if payload:
                    try:
                        arr = np.asarray(payload["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"], dtype=np.float64)
                        fps = row["detail"]["video_metadata"]["average_fps"]
                        w = row["detail"]["video_metadata"]["width"]
                        h = row["detail"]["video_metadata"]["height"]
                        n = round((record["end_time"] - record["start_time"]) * 25)
                        indices = np.asarray([round((record["start_time"] + i / 25) * fps) for i in range(n)], dtype=np.int64)
                        coord = arr[indices]
                        valid = np.isfinite(coord).all(axis=1) & (coord[:, 0] >= 0) & (coord[:, 1] >= 0) & (coord[:, 2] <= h) & (coord[:, 3] <= w) & (coord[:, 0] < coord[:, 2]) & (coord[:, 1] < coord[:, 3])
                        res["invalid_lip_boxes_in_nominal_selected_frames"] = int((~valid).sum())
                        res["invalid_lip_frame_indices"] = indices[~valid].tolist()
                        for code in ("Lip_bounding_box_invalid_geometry", "Lip_bounding_box_outside_image_yxyx"):
                            if code in errs and valid.all():
                                errs.remove(code)
                    except Exception as exc:
                        res["selected_box_check_exception"] = repr(exc)
                res["errors"] = sorted(set(errs))
                results.append(res)
    finally:
        for tf in archives.values():
            if tf:
                tf.close()
    document = {"scope": "Frozen legacy B512 and new stage18 pilot120 against D: labels; nominal sentence windows, no new audio alignment inference",
                "groups": {group: {"selected_sentences": sum(r["group"] == group for r in results),
                    "sentences_with_errors": sum(r["group"] == group and bool(r["errors"]) for r in results),
                    "frozen_label_hash_mismatches": sum(r["group"] == group and r.get("label_sha256_matches_frozen_selection") is False for r in results)}
                    for group in sorted({r["group"] for r in results})}, "records": results}
    save(out / "selection_impact.json", document)
    print(json.dumps(document["groups"], ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    synthetic_tests()
    if args.self_test:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    metadata, package_media, archives, metadata_errors = {}, {}, [], []
    for p in sorted(args.root.rglob("TS*.tar")):
        before = p.stat()
        video_names, audio_names, duplicates = set(), set(), []
        with tarfile.open(p, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = PurePosixPath(m.name).name
                if name.endswith(".wav"):
                    if name in audio_names:
                        duplicates.append(name)
                    audio_names.add(name)
                elif name.endswith(".mp4"):
                    if name in video_names or name in metadata:
                        duplicates.append(name)
                    video_names.add(name)
                    try:
                        with tf.extractfile(m) as f:
                            metadata[name] = video_metadata(f, m.size)
                    except Exception as exc:
                        metadata_errors.append({"archive": str(p), "member": m.name, "error": str(exc)})
        after = p.stat()
        package_media[p.stem.replace("TS", "TL", 1)] = (video_names, audio_names)
        archives.append({"path": str(p), "size": before.st_size, "mtime_ns": before.st_mtime_ns,
                         "unchanged_during_read": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                         "video_count": len(video_names), "audio_count": len(audio_names), "duplicate_names": duplicates})
        print(f"media_metadata {p.name}: videos={len(video_names)} audio={len(audio_names)} errors={len(metadata_errors)}", flush=True)
    save(args.out / "video_metadata.json", {"videos": metadata, "errors": metadata_errors, "archives": archives})
    rows, package_stats, name_seen, grouped = [], [], set(), defaultdict(list)
    for p in sorted(args.root.rglob("TL*.tar")):
        before = p.stat()
        stats = {"archive": str(p), "size": before.st_size, "mtime_ns": before.st_mtime_ns}
        videos, audios = package_media.get(p.stem, (set(), set()))
        local_names = set()
        with tarfile.open(p, "r:") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                stem = PurePosixPath(m.name).stem
                row = {"archive": str(p), "member": m.name, "stem": stem, "size": m.size}
                with tf.extractfile(m) as f:
                    payload = f.read()
                row["sha256"] = hashlib.sha256(payload).hexdigest()
                try:
                    data = json.loads(payload.decode("utf-8-sig"))
                    row.update(validate_label(data, stem, metadata.get(stem + ".mp4"), videos, audios))
                except Exception as exc:
                    row.update(errors=["json_parse_or_validation_exception"], warnings=[], detail={"exception": repr(exc)})
                if stem in name_seen:
                    row["errors"].append("duplicate_label_basename")
                name_seen.add(stem)
                local_names.add(stem + ".mp4")
                ident = row.get("detail", {}).get("identity")
                if ident:
                    grouped[ident["recording"]].append(row)
                rows.append(row)
        after = p.stat()
        stats["unchanged_during_read"] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        stats["videos_without_label"] = sorted(videos - local_names)
        stats["labels_without_video"] = sorted(local_names - videos)
        package_stats.append(stats)
        save(args.out / "label_records.json", rows)
        print(f"labels {p.name}: count={len(local_names)} cumulative_errors={sum(bool(r['errors']) for r in rows)}", flush=True)
    group_issues = []
    for recording, rr in grouped.items():
        angles = sorted(r["detail"]["identity"]["angle"] for r in rr)
        sigs = {r["detail"].get("sentence_signature_sha256") for r in rr}
        if len(sigs) != 1 or len(angles) != 5 or "A" not in angles:
            group_issues.append({"recording": recording, "angles": angles, "sentence_signature_count": len(sigs)})
    errors = Counter(e for r in rows for e in r["errors"])
    warnings = Counter(e for r in rows for e in r["warnings"])
    summary = {"audit_date": "2026-10-02", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All JSON labels in five local TL TARs, MP4 moov metadata in matching TS TARs; no payload decode or transcript truth assessment",
        "label_count": len(rows), "labels_with_errors": sum(bool(r["errors"]) for r in rows),
        "labels_with_warnings": sum(bool(r["warnings"]) for r in rows),
        "frontal_labels": sum(r.get("detail", {}).get("identity", {}).get("angle") == "A" for r in rows),
        "frontal_labels_with_errors": sum(bool(r["errors"]) and r.get("detail", {}).get("identity", {}).get("angle") == "A" for r in rows),
        "sentence_annotations_all_angles": sum(r.get("detail", {}).get("sentence_count", 0) for r in rows),
        "sentence_annotations_frontal": sum(r.get("detail", {}).get("sentence_count", 0) for r in rows if r.get("detail", {}).get("identity", {}).get("angle") == "A"),
        "recording_count": len(grouped), "error_counts": dict(errors), "warning_counts": dict(warnings),
        "angles": dict(Counter(r.get("detail", {}).get("identity", {}).get("angle") for r in rows)),
        "speakers": sorted({r.get("detail", {}).get("identity", {}).get("speaker") for r in rows if r.get("detail", {}).get("identity")}),
        "package_stats": package_stats, "cross_angle_group_issues": group_issues,
        "video_metadata_errors": metadata_errors, "elapsed_seconds": time.monotonic() - start,
        "limitations": ["No official publisher checksum comparison.", "A parsed label and valid metadata do not prove the transcript matches speech or lip movements.", "Nonfrontal labels declare nonexistent same-angle WAVs; A-angle WAV is a candidate shared recording, not assumed time-aligned.", "Bounding boxes use existing project's visually established yxyx convention; no new manual crop review.", "MP4 sample counts are encoded metadata; successful decoding is a separate audit."]}
    save(args.out / "summary.json", summary)
    save(args.out / "label_records.json", rows)
    print(json.dumps({k: summary[k] for k in ("label_count", "labels_with_errors", "frontal_labels_with_errors", "error_counts", "warning_counts", "elapsed_seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
