"""Read-only label/time and saved-output audit; never fixes targets or timestamps.

Reads two frozen C025 JSON labels and only MP4 headers/moov metadata. No mdat
payload, model execution, audio decoding, training or source writes are used.
Similarity to neighboring transcripts is diagnostic, not ground-truth validation.
"""
from __future__ import annotations

import hashlib
import json
import struct
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_normalize(text: str) -> str:
    # Preserve unsupported labels such as X; punctuation -> space, as for project GT.
    chars = [" " if unicodedata.category(ch).startswith("P") else ch
             for ch in unicodedata.normalize("NFC", text)]
    return " ".join("".join(chars).split())


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def boxes(data: bytes):
    position = 0
    while position < len(data):
        if len(data) - position < 8:
            raise ValueError("Incomplete box header")
        size, name = struct.unpack_from(">I4s", data, position)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, position + 8)[0]
            header = 16
        elif size == 0:
            size = len(data) - position
        if size < header or position + size > len(data):
            raise ValueError("Invalid box size")
        yield name.decode("ascii", errors="replace"), data[position+header:position+size]
        position += size


def timed_header(payload: bytes) -> dict:
    version = payload[0]
    if version == 0:
        scale, ticks = struct.unpack_from(">II", payload, 12)
    elif version == 1:
        scale = struct.unpack_from(">I", payload, 20)[0]
        ticks = struct.unpack_from(">Q", payload, 24)[0]
    else:
        raise ValueError("Unknown time header version")
    return {"timescale": scale, "duration_ticks": ticks, "duration_seconds": ticks/scale}


def mp4_metadata(path: Path) -> dict:
    size = path.stat().st_size
    result = {"path": str(path), "size_bytes": size, "mdat_payload_read": False,
              "audio_payload_read": False, "header_bytes_read": 0, "tracks": []}
    with path.open("rb") as stream:
        offset = 0
        moov = None
        while offset < size:
            stream.seek(offset)
            raw = stream.read(8)
            result["header_bytes_read"] += len(raw)
            length, name = struct.unpack(">I4s", raw)
            header = 8
            if length == 1:
                raw = stream.read(8)
                result["header_bytes_read"] += len(raw)
                length, header = struct.unpack(">Q", raw)[0], 16
            elif length == 0:
                length = size - offset
            if length < header or offset + length > size:
                raise ValueError("Invalid top-level box")
            if name == b"moov":
                if length > 10_000_000:
                    raise ValueError("Unexpectedly large metadata")
                moov = stream.read(length-header)
                result["header_bytes_read"] += len(moov)
            offset += length
    if moov is None:
        raise ValueError("Missing moov")
    top = list(boxes(moov))
    result["movie"] = timed_header(next(data for name, data in top if name == "mvhd"))
    for name, payload in top:
        if name != "trak":
            continue
        track = {}
        children = dict(boxes(payload))
        mdia = dict(boxes(children["mdia"]))
        track["handler_type"] = mdia["hdlr"][8:12].decode("ascii")
        track["media"] = timed_header(mdia["mdhd"])
        if "edts" in children:
            edts = dict(boxes(children["edts"]))
            if "elst" in edts:
                elst = edts["elst"]
                count = struct.unpack_from(">I", elst, 4)[0]
                entries, pos = [], 8
                for _ in range(count):
                    if elst[0] == 0:
                        duration, media_time, rate_i, rate_f = struct.unpack_from(">Iihh", elst, pos)
                        pos += 12
                    else:
                        duration, media_time, rate_i, rate_f = struct.unpack_from(">Qqhh", elst, pos)
                        pos += 20
                    entries.append({"segment_movie_ticks": duration, "media_start_ticks": media_time,
                                    "segment_seconds": duration/result["movie"]["timescale"],
                                    "media_start_seconds": media_time/track["media"]["timescale"],
                                    "rate_integer": rate_i, "rate_fraction": rate_f})
                track["edit_list"] = entries
        result["tracks"].append(track)
    result["audio_track_present"] = any(t["handler_type"] == "soun" for t in result["tracks"])
    return result


def candidate(row: dict, prediction: str, target: dict) -> dict:
    text = text_normalize(row["sentence_text"])
    edits = distance(text, prediction)
    return {"sentence_id": int(row["ID"]), "sentence_id_delta": int(row["ID"]) - int(target["ID"]),
            "source_text": row["sentence_text"], "normalized_text": text,
            "start_time": row["start_time"], "end_time": row["end_time"],
            "start_delta_seconds": row["start_time"] - target["start_time"],
            "edits": edits, "reference_characters": len(text),
            "cer": edits/len(text),
            "max_length_normalized_edit_distance": edits/max(len(text), len(prediction), 1)}


def main() -> None:
    result_path = BASE / "results/step04_C025_alignment.json"
    if result_path.exists():
        raise FileExistsError("Audit output exists; do not overwrite evidence")
    plan_path, result03_path = BASE/"step03_source_plan.json", BASE/"results/step03.json"
    manifest_path = BASE/"inputs_step03_recovery02/manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    predictions = json.loads(result03_path.read_text(encoding="utf-8"))
    prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = {r["id"]: r for r in plan["records"] if r["speaker_id"] == "C025"}
    sources, labels = [], {}
    for source in plan["sources"]:
        if source["source_video"]["speaker_id"] != "C025":
            continue
        label_path = Path(source["label_inventory"]["path"])
        data = label_path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != source["label_sha256"]:
            raise ValueError("Label changed")
        label = json.loads(data.decode("utf-8-sig"))[0]
        rows = label["Sentence_info"]
        labels[source["stem"]] = rows
        source_video = Path(source["source_video"]["path"])
        native = prepared["sources"][source_video.name]
        duration = native["reported_frames"] / native["native_fps"]
        bbox = label["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        sources.append({"stem": source["stem"], "label_path": str(label_path),
                        "label_sha256": actual_hash,
                        "video_info": label["Video_info"], "audio_info": label["Audio_info"],
                        "mp4": mp4_metadata(source_video),
                        "native_fps": native["native_fps"], "native_frame_count": native["reported_frames"],
                        "native_frame_duration_seconds": duration,
                        "bbox_count": len(bbox), "bbox_frame_index_span": [0, len(bbox)-1],
                        "bbox_last_index_native_time_seconds": (len(bbox)-1)/native["native_fps"],
                        "bbox_first_entry": bbox[0], "bbox_last_entry": bbox[-1],
                        "label_sentence_count": len(rows),
                        "label_first_start_seconds": min(r["start_time"] for r in rows),
                        "label_last_end_seconds": max(r["end_time"] for r in rows),
                        "sentence_ids_extending_beyond_native_duration": [r["ID"] for r in rows if r["end_time"] > duration],
                        "overrun_seconds": max(r["end_time"] for r in rows)-duration,
                        "sentences": rows})
    records = []
    for pred in predictions["records"]:
        if pred["id"] not in selected:
            continue
        row = selected[pred["id"]]
        stem = pred["id"].rsplit("_", 1)[0]
        rows = labels[stem]
        target = next(r for r in rows if int(r["ID"]) == row["sentence_id"])
        if text_normalize(target["sentence_text"]) != pred["reference"]:
            raise ValueError("Target mismatch")
        methods = {}
        for method, output in pred["outputs"].items():
            prediction = text_normalize(output["prediction"])
            candidates = [candidate(r, prediction, target) for r in rows]
            ranked = sorted(candidates, key=lambda r: (r["max_length_normalized_edit_distance"], r["edits"], r["sentence_id"]))
            target_score = next(r for r in ranked if r["sentence_id"] == target["ID"])
            methods[method] = {"status": output["status"], "prediction": output["prediction"],
                               "target_rank": next(i for i,r in enumerate(ranked,1) if r["sentence_id"] == target["ID"]),
                               "target_score": target_score, "best_three_same_recording": ranked[:3],
                               "nearby_candidates": [r for r in candidates if abs(r["sentence_id_delta"]) <= 2],
                               "best_by_raw_cer": min(candidates, key=lambda r:r["cer"])}
        records.append({"id": pred["id"], "arm": pred["arm"], "reference": pred["reference"], "methods": methods})
    groups = {}
    for arm in ("annotation_bbox", "automatic_mediapipe"):
        for method in ("corrected_hybrid", "ctc_greedy"):
            rows = [r["methods"][method] for r in records if r["arm"] == arm]
            groups[f"{arm}/{method}"] = {
                "n": len(rows),
                "best_matching_sentence_delta_counts": dict(Counter(str(r["best_three_same_recording"][0]["sentence_id_delta"]) for r in rows)),
                "next_sentence_better_than_target": sum(next(n for n in r["nearby_candidates"] if n["sentence_id_delta"] == 1)["max_length_normalized_edit_distance"] < r["target_score"]["max_length_normalized_edit_distance"] for r in rows),
                "target_best_matching_count": sum(r["target_rank"] == 1 for r in rows)}
    result = {"schema": "olkavs-step04-c025-content-alignment-audit-v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(Path(__file__)), "source_plan_sha256": sha(plan_path),
              "step03_result_sha256": sha(result03_path), "prepared_manifest_sha256": sha(manifest_path),
              "scope": "two C025 labels, MP4 headers/moov only, and saved step03 predictions",
              "sources": sources, "records": records, "summary": groups,
              "method": "Rank each saved output against every sentence in the same recording by Levenshtein distance/max(reference_length,prediction_length); also record raw CER and +/-2 neighboring labels. Text NFC, punctuation to space, collapsed whitespace; X is preserved.",
              "conclusion": "Repeated resemblance to later annotated sentences plus labels exceeding actual video duration raises a substantial timeline-alignment concern. Metadata name/hash agreement did not establish speech-content alignment.",
              "limits": ["No actual speech listened to or transcribed; no independent truth validation.",
                         "Outputs are fallible and cannot justify changing references or selecting an optimal time offset.",
                         "Nominal duration or terminal overrun alone cannot identify whether a file was trimmed, a label is wrong, or an offset belongs to another media track.",
                         "No label, timestamp, crop, model or dataset was changed.",
                         "No test50 payload or new inference was used."]}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"path":str(result_path),"sha256":sha(result_path),"summary":groups,
                      "source_timing":[{k:s[k] for k in ("stem","native_frame_duration_seconds","label_last_end_seconds","overrun_seconds","sentence_ids_extending_beyond_native_duration","mp4")} for s in sources]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
