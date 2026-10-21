"""Re-read only time-error labels to retain exact offending intervals."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

base = Path(__file__).resolve().parent
rows = json.loads((base / "label_records.json").read_text(encoding="utf-8"))
by_archive = defaultdict(list)
for row in rows:
    if row["errors"]:
        by_archive[row["archive"]].append(row)
details = []
for archive, rr in by_archive.items():
    with tarfile.open(archive, "r:") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for row in rr:
            with tf.extractfile(members[row["member"]]) as f:
                raw = f.read()
            assert hashlib.sha256(raw).hexdigest() == row["sha256"], row["stem"]
            obj = json.loads(raw.decode("utf-8-sig"))
            obj = obj[0] if isinstance(obj, list) else obj
            duration = row["detail"]["video_metadata"]["duration"]
            bad = []
            for issue in row["detail"].get("sentence_issues", []):
                sent = obj["Sentence_info"][issue["ordinal"]]
                assert sent["ID"] == issue["ID"]
                bad.append({"sentence_id": sent["ID"], "start_time": sent["start_time"],
                    "end_time": sent["end_time"], "video_duration": duration,
                    "end_past_video_seconds": sent["end_time"] - duration,
                    "issues": issue["issues"]})
            details.append({"stem": row["stem"], "identity": row["detail"]["identity"],
                "archive": archive, "member": row["member"], "label_sha256": row["sha256"],
                "declared_video_duration": obj["Video_info"]["video_Duration"],
                "actual_video_sample_duration": duration, "time_issues": bad})
result = {"scope": "Exact original label intervals versus encoded MP4 video-track duration; no presumed offset corrections",
          "label_count": len(details), "affected_sentence_annotations": sum(len(r["time_issues"]) for r in details),
          "labels_by_speaker": dict(Counter(r["identity"]["speaker"] for r in details)),
          "records": details, "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(base / "time_anomaly_details.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
for row in details:
    if row["identity"]["speaker"] != "C025":
        print(row["stem"], [(r["sentence_id"], round(r["end_past_video_seconds"], 6)) for r in row["time_issues"]])
print("saved_time_anomaly_details", result["label_count"], result["affected_sentence_annotations"])
