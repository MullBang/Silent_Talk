"""Read-only project audit; model inference is stubbed for protocol reproduction.

Run with a Python runtime containing NumPy. Does not load or change model weights.
Writes only audit_evidence.json alongside this script.
"""
from __future__ import annotations

import ast
import asyncio
import ctypes
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from services import ws_handler as ws
from models import GRAPHEME_VOCAB, IDX_TO_GRAPHEME, VALID_TAILS, is_consonant, is_vowel


class FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


async def protocol_case(final_frames):
    sock = FakeSocket()
    state = {"session_id": None, "client_ts": None, "chunk": None}
    buffers = {}
    calls = []

    def fake_infer(frames, with_confidence):
        calls.append({"frames": len(frames), "final": with_confidence})
        return f"frames={len(frames)}", 0.5 if with_confidence else None

    ws._infer = fake_infer

    async def control(**message):
        await ws.handle_control_message(sock, json.dumps(message), state, buffers)

    async def frame(chunk):
        packet = chunk.to_bytes(4, "big") + b"\x01\x00\x00\x00" + bytes(ws.EXPECTED_ROI_BYTES)
        await ws.handle_binary_chunk(sock, packet, state, buffers)

    await control(type="SESSION_START", session_id="audit", client_ts=0)
    await control(type="CHUNK_START", chunk_id=1, is_final=False)
    for _ in range(11):
        await frame(1)
    await control(type="CHUNK_START", chunk_id=2, is_final=True)
    for _ in range(final_frames):
        await frame(2)
    return {
        "sent_frames": 11 + final_frames,
        "inference_calls": calls,
        "final_messages": [m for m in sock.messages if m.get("is_final")],
        "errors": [m for m in sock.messages if m["type"] == "ERROR"],
        "remaining_buffer_frames": len(buffers["audit"]),
    }


def npy_header(archive, name):
    with archive.open(name) as stream:
        major, minor = np.lib.format.read_magic(stream)
        if (major, minor) == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)
        if (major, minor) == (2, 0):
            return np.lib.format.read_array_header_2_0(stream)
        raise ValueError(f"Unsupported npy header: {major}.{minor}")


def inventory():
    groups = []
    for folder in sorted((ROOT / "data_cache").iterdir()):
        files = sorted(folder.glob("*.npz")) if folder.is_dir() else []
        if not files:
            continue
        texts, lengths, speakers, views = [], [], set(), set()
        for path in files:
            with zipfile.ZipFile(path) as archive:
                shape, _, dtype = npy_header(archive, "frames.npy")
            with np.load(path, allow_pickle=False) as d:
                texts.append(str(d["text"]))
            lengths.append(shape[0])
            match = re.search(r"_([A-Z]\d{3})_([A-I])_", path.name)
            if match:
                speakers.add(match[1]); views.add(match[2])
            assert tuple(shape[1:]) == (96, 96, 3) and dtype == np.uint8
        groups.append({"folder": folder.name, "files": len(files),
                       "unique_texts": len(set(texts)), "T_min": min(lengths), "T_max": max(lengths),
                       "seconds_at_25fps": sum(lengths) / 25,
                       "compressed_bytes": sum(p.stat().st_size for p in files),
                       "speaker_ids_from_filenames": sorted(speakers), "views": sorted(views)})
    return groups


def split_check():
    files = sorted((ROOT / "data_cache/clips").glob("*.npz"))
    n_val = int(len(files) * 0.15)
    train, val = files[:-n_val], files[-n_val:]
    def groups(items):
        return {re.sub(r"_([A-I])_(\d{3}_\d{3})$", r"_\2", p.stem) for p in items}
    return {"train_files": len(train), "val_files": len(val),
            "same_speaker": "C002", "same_recording_utterance_ids_across_views": len(groups(train) & groups(val))}


def label_check():
    # Unicode composition substitutes only jamo.j2h, enabling the unchanged
    # compose_jamos function to run without installing the ML environment.
    leads = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    vowels = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    tails = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
    def j2h(lead, vowel, tail=""):
        return chr(0xAC00 + (leads.index(lead) * 21 + vowels.index(vowel)) * 28 + tails.index(tail))
    tree = ast.parse((ROOT / "backend/models/decoder.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compose_jamos")
    scope = {"j2h": j2h, "is_consonant": is_consonant, "is_vowel": is_vowel, "VALID_TAILS": VALID_TAILS}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "compose_jamos-extracted", "exec"), scope)
    samples = []
    for suffix in ["002", "003"]:
        p = ROOT / f"data_cache/test_all51/lip_J_1_F_02_C002_A_001_{suffix}.npz"
        with np.load(p, allow_pickle=False) as d:
            graphemes = [IDX_TO_GRAPHEME[int(i)] for i in d["label"]]
            samples.append({"file": str(p.relative_to(ROOT)), "reference": str(d["text"]),
                            "decoded_saved_target_labels": scope["compose_jamos"](graphemes)})
    return {"missing_compound_tails": [c for c in tails if c and c not in GRAPHEME_VOCAB],
            "space_in_vocabulary": " " in GRAPHEME_VOCAB,
            "method": "Actual stored labels and unchanged decoder logic; Unicode j2h substitute; no neural inference.",
            "samples": samples}


def main():
    multi = asyncio.run(protocol_case(3))
    empty = asyncio.run(protocol_case(0))
    assert multi["final_messages"][0]["text"] == "frames=12"
    assert len(multi["errors"]) == 2 and multi["sent_frames"] == 14
    assert not empty["final_messages"] and empty["remaining_buffer_frames"] == 11
    params = {"conv_and_bn": 327264, "lstm_layer1": 113774592,
              "lstm_layer2": 1576960, "linear": 21033}
    total = sum(params.values())
    evidence = {
        "date": "2026-09-17", "root": str(ROOT),
        "scope": "Static source inspection, NPZ metadata and target inspection, isolated protocol reproduction; no training or end-to-end webcam benchmark.",
        "cache": inventory(), "default_split": split_check(), "label_check": label_check(),
        "protocol_final_three_frames": multi, "protocol_empty_final": empty,
        "parameter_count_from_layer_shapes": {**params, "total": total,
            "fp32_weights_MiB": total * 4 / 2**20,
            "fp32_weights_grad_adam_moments_GiB_excluding_activations": total * 16 / 2**30},
        "checkpoint": {"path": "backend/models/weights/baseline.pt", "bytes": (ROOT / "backend/models/weights/baseline.pt").stat().st_size},
        "storage_calculations_decimal_GB": {"roi_5seconds_RGB_uint8_MB": 25 * 5 * 96 * 96 * 3 / 1e6,
            "roi_100000_5second_clips_GB": 25 * 5 * 96 * 96 * 3 * 100000 / 1e9,
            "roi_1150hours_GB": 25 * 3600 * 1150 * 96 * 96 * 3 / 1e9,
            "raw_1080p_30fps_180seconds_GiB": 1920 * 1080 * 3 * 30 * 180 / 2**30},
        "machine_observed": {"cpu": "AMD Ryzen 7 5800X 8-Core Processor", "gpu": "NVIDIA GeForce RTX 3080",
            "gpu_memory_MiB": 10240, "ram_GiB": 31.95, "C_free_bytes_approx": 241098715136, "D_free_bytes_approx": 144234348544},
    }
    target = Path(__file__).with_name("audit_evidence.json")
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
