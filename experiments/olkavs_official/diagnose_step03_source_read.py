"""Read-only decoder recovery probe for the two already-selected C294 sources."""
import json
import time
from pathlib import Path

import cv2

from prepare_step03_inputs import native_bounds, sha256

BASE=Path(__file__).resolve().parent
plan=json.loads((BASE/"step03_source_plan.json").read_text(encoding="utf-8"))
out=BASE/"step03_source_read_probe.json"
if out.exists():
    raise FileExistsError(out)
report={"opencv":cv2.__version__,"plan_sha256":sha256(BASE/"step03_source_plan.json"),
        "model_run":False,"source_modified":False,"sources":[]}
for stem in sorted({row["source_stem"] for row in plan["records"] if row["speaker_id"]=="C294"}):
    rows=[row for row in plan["records"] if row["source_stem"]==stem]
    path=Path(rows[0]["source_video"]["path"])
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    source={"path":str(path),"native_fps":fps,"frames":cap.get(cv2.CAP_PROP_FRAME_COUNT),
            "width":cap.get(cv2.CAP_PROP_FRAME_WIDTH),"height":cap.get(cv2.CAP_PROP_FRAME_HEIGHT),"probes":[]}
    cap.release()
    for row in rows:
        first,stop=native_bounds(row["start_time"],row["end_time"],fps)
        cap=cv2.VideoCapture(str(path))
        started=time.perf_counter()
        seek_ok=cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        seek_pos=cap.get(cv2.CAP_PROP_POS_FRAMES)
        decoded=0
        for idx in range(first,stop):
            ok,frame=cap.read()
            if not ok:
                break
            decoded+=1
        result={"id":row["id"],"first":first,"stop":stop,"seek_ok":seek_ok,
                "seek_position":seek_pos,"decoded":decoded,"needed":stop-first,
                "final_position":cap.get(cv2.CAP_PROP_POS_FRAMES),
                "seconds":time.perf_counter()-started}
        cap.release()
        source["probes"].append(result)
        print(json.dumps(result),flush=True)
    report["sources"].append(source)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(str(out),flush=True)
