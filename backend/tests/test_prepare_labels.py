"""영상 추출만 대체하고 신규 데이터 준비의 라벨 저장 계약을 검사한다."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
import prepare_data
from models.cache import load_clip


def test_prepare_saves_versioned_lossless_labels(tmp_path, monkeypatch):
    label = tmp_path / "label.json"
    label.write_text(json.dumps({"Sentence_info": [
        {"ID": 1, "sentence_text": "값이 없었어.", "start_time": 0, "end_time": 2}
    ]}, ensure_ascii=False), encoding="utf-8")
    frames = np.zeros((50, 96, 96, 3), dtype=np.uint8)
    monkeypatch.setattr(prepare_data, "_cut_sentence_clip", lambda *a: True)
    monkeypatch.setattr(prepare_data, "resample_to_fps", lambda *a: (list(frames), 25))
    monkeypatch.setattr(prepare_data, "extract_roi_from_segment", lambda x: x)
    out = tmp_path / "cache"
    assert prepare_data.prepare_video("video.mp4", str(label), str(out)) == 1
    clip = load_clip(out / "video_001.npz")
    assert clip["text"] == "값이 없었어"
    assert clip["source_text"] == "값이 없었어."


def test_unknown_text_fails_before_video_processing(tmp_path, monkeypatch):
    label = tmp_path / "label.json"
    label.write_text(json.dumps({"Sentence_info": [
        {"ID": 1, "sentence_text": "안녕"}, {"ID": 2, "sentence_text": "X 이유"}
    ]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(prepare_data, "_cut_sentence_clip", lambda *a: pytest.fail("영상 처리 전에 거부해야 함"))
    with pytest.raises(ValueError, match="ID=2"):
        prepare_data.prepare_video("video.mp4", str(label), str(tmp_path / "cache"))
    assert not list((tmp_path / "cache").glob("*.npz"))


def test_selection_skips_unselected_invalid_text_and_preserves_provenance(tmp_path, monkeypatch):
    label = tmp_path / "label.json"
    label.write_text(json.dumps({"Sentence_info": [
        {"ID": 1, "sentence_text": "X 이유"},
        {"ID": 2, "sentence_text": "값이 없다.", "start_time": 0, "end_time": 2},
    ]}, ensure_ascii=False), encoding="utf-8")
    frames = np.zeros((50, 96, 96, 3), dtype=np.uint8)
    monkeypatch.setattr(prepare_data, "_cut_sentence_clip", lambda *a: True)
    monkeypatch.setattr(prepare_data, "resample_to_fps", lambda *a: (list(frames), 25))
    monkeypatch.setattr(prepare_data, "extract_roi_from_segment", lambda x: x)
    out = tmp_path / "cache"
    assert prepare_data.prepare_video("video.mp4", str(label), str(out), sentence_ids=[2],
                                      provenance={2: {"speaker_id": "C002"}}) == 1
    assert not (out / "video_001.npz").exists()
    with np.load(out / "video_002.npz", allow_pickle=False) as clip:
        assert str(clip["speaker_id"]) == "C002"
    before = (out / "video_002.npz").read_bytes()
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        prepare_data.prepare_video("video.mp4", str(label), str(out), sentence_ids=[2])
    assert (out / "video_002.npz").read_bytes() == before


def test_unknown_selected_sentence_is_rejected(tmp_path):
    label = tmp_path / "label.json"
    label.write_text(json.dumps({"Sentence_info": [{"ID": 1, "sentence_text": "안녕"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="존재하지 않는"):
        prepare_data.prepare_video("video.mp4", str(label), str(tmp_path / "out"), sentence_ids=[99])


class ArrayCapture:
    """지정 배열을 원본 BGR 영상 프레임처럼 읽는 테스트 캡처."""

    def __init__(self, frame, available=1, fps=25, declared_count=None, pts_offset=0):
        self.frame, self.available, self.index = frame, available, 0
        self.fps = fps
        self.declared_count = available if declared_count is None else declared_count
        self.pts_offset = pts_offset
        self.seek_indices = []

    def set(self, prop, index):
        self.index = int(index)
        self.seek_indices.append(int(index))

    def get(self, prop):
        if prop == prepare_data.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == prepare_data.cv2.CAP_PROP_FRAME_COUNT:
            return self.declared_count
        if prop == prepare_data.cv2.CAP_PROP_POS_FRAMES:
            return self.index
        if prop == prepare_data.cv2.CAP_PROP_POS_MSEC:
            return ((self.index - 1) / self.fps + self.pts_offset) * 1000
        raise AssertionError(prop)

    def release(self):
        pass

    def read(self):
        self.index += 1
        return (True, self.frame.copy()) if self.index <= self.available else (False, None)


def test_yxyx_boxes_crop_upright_lips_without_rotating_frame():
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # blue background
    frame[20:40, 60:80] = (0, 0, 255)  # red target, offset to detect swapped axes
    label = {"Video_info": {"FPS": 25}, "Bounding_box_info": {
        "Lip_bounding_box": {"xtl_ytl_xbr_ybr": [[20, 60, 40, 80]]}}}
    boxes, fps = prepare_data._extract_lip_boxes(label)
    assert boxes == [[60, 20, 80, 40]]
    roi = prepare_data._roi_seq_from_boxes(ArrayCapture(frame), 0, 0.04, boxes, fps)
    assert roi.shape == (1, 96, 96, 3)
    assert roi[0, 48, 48].tolist() == [255, 0, 0]  # RGB red target


def test_xyxy_order_can_be_explicitly_selected():
    label = {"Bounding_box_info": {"Lip_bounding_box": {"xtl_ytl_xbr_ybr": [[60, 20, 80, 40]]}}}
    assert prepare_data._extract_lip_boxes(label, "xyxy")[0] == [[60, 20, 80, 40]]


def test_truncated_video_does_not_reuse_stale_frame():
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    assert prepare_data._roi_seq_from_boxes(ArrayCapture(frame, 1, declared_count=2), 0, 0.08,
                                            [[60, 20, 80, 40]] * 2, 25) is None


def test_out_of_bounds_box_is_rejected_instead_of_cropping_background():
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="범위 밖"):
        prepare_data._roi_seq_from_boxes(ArrayCapture(frame), 0, 0.04, [[20, 60, 40, 80]], 25)


def test_video_2997_fps_not_rounded_label_30_controls_late_sentence_seek():
    """300초에서 라벨30fps를 쓰면 약0.3초/9프레임 밀리는 실제 회귀를 방지한다."""
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    video_fps = 30000 / 1001
    capture = ArrayCapture(frame, available=9053, fps=video_fps)
    rois = prepare_data._roi_seq_from_boxes(capture, 300.0, 301.0,
                                           [[60, 20, 80, 40]] * 9053, 30)
    assert rois.shape == (25, 96, 96, 3)
    assert capture.seek_indices == [8991]
    assert abs(capture.seek_indices[0] / video_fps - 300.0) < .5 / video_fps
    assert round(300.0 * 30) - capture.seek_indices[0] == 9


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_actual_fps_is_rejected_without_label_fps_fallback(fps):
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="실제 FPS"):
        prepare_data._roi_seq_from_boxes(ArrayCapture(frame, fps=fps), 0, .04,
                                        [[60, 20, 80, 40]], 30)


def test_video_frame_and_bbox_count_must_match():
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="bbox 수"):
        prepare_data._roi_seq_from_boxes(ArrayCapture(frame, available=2), 0, .04,
                                        [[60, 20, 80, 40]], 30)


@pytest.mark.parametrize("pts_offset", [.005, .04, float("nan")])
def test_missing_or_non_cfr_pts_is_rejected(pts_offset):
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="PTS"):
        prepare_data._roi_seq_from_boxes(ArrayCapture(frame, pts_offset=pts_offset), 0, .04,
                                        [[60, 20, 80, 40]], 30)


def test_wrong_seek_result_is_rejected():
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    capture = ArrayCapture(frame, available=2)
    capture.set = lambda prop, index: setattr(capture, "index", int(index) + 1)
    with pytest.raises(ValueError, match="시크 결과"):
        prepare_data._roi_seq_from_boxes(capture, 0, .04, [[60, 20, 80, 40]] * 2, 30)


@pytest.mark.parametrize("start,end", [(-.01,.02), (0,.041), (.02,.02), (0,float("nan"))])
def test_sentence_bounds_must_fit_actual_video_duration(start, end):
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="실제 재생 범위"):
        prepare_data._roi_seq_from_boxes(ArrayCapture(frame), start, end,
                                        [[60, 20, 80, 40]], 30)


def test_bbox_cache_records_actual_and_label_fps_separately(tmp_path, monkeypatch):
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    capture = ArrayCapture(frame, available=50, fps=25)
    monkeypatch.setattr(prepare_data.cv2, "VideoCapture", lambda path: capture)
    label = {"Video_info": {"FPS": 30}, "Bounding_box_info": {
        "Lip_bounding_box": {"xtl_ytl_xbr_ybr": [[20, 60, 40, 80]] * 50}},
        "Sentence_info": [{"ID": 1, "sentence_text": "안녕", "start_time": 0, "end_time": 2}]}
    label_path = tmp_path / "label.json"
    label_path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    assert prepare_data.prepare_video("clip.mp4", str(label_path), str(out), roi_mode="bbox") == 1
    with np.load(out / "clip_001.npz", allow_pickle=False) as data:
        assert str(data["temporal_method"]) == "video-cfr-timestamps-v1"
        assert float(data["video_fps"]) == 25
        assert float(data["label_fps"]) == 30
