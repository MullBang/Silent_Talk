"""실제 WebSocket 전송·프레임 경계·세션 종료 회귀 검사. 모델만 대체한다."""
import asyncio
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CORS_ORIGINS, WS_MAX_UTTERANCE_FRAMES
from main import app
from services import ws_handler as handler

client = TestClient(app)
ROI_BYTES = handler.EXPECTED_ROI_BYTES


def frame(chunk_id, count=ROI_BYTES, msg_type=1, fill=128):
    return chunk_id.to_bytes(4, 'big') + bytes([msg_type, 0, 0, 0]) + bytes([fill]) * count


def start(ws):
    ws.send_json({'type': 'SESSION_START', 'protocol_version': 2, 'session_id': 's1', 'client_ts': 1})
    ready = ws.receive_json()
    assert ready['type'] == 'SESSION_READY'
    assert ready['fps'] == 25


def chunk(ws, cid=1, uid=1, count=1, begin=0, final=False):
    ws.send_json({'type': 'CHUNK_START', 'chunk_id': cid, 'utterance_id': uid,
                  'frame_count': count, 'window_start_ms': begin,
                  'window_end_ms': begin + count * 40, 'is_final': final})


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(handler, '_get_model', lambda: object())
    calls = []
    def infer(frames, with_confidence):
        calls.append(([int(f[0, 0, 0]) for f in frames], with_confidence))
        return ('안녕' if with_confidence else '안', 0.91 if with_confidence else None)
    monkeypatch.setattr(handler, '_infer', infer)
    return calls


def test_multi_frame_final_waits_for_every_frame_and_predicts_once(fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws)
        chunk(ws, count=3, final=True)
        for value in (10, 20, 30): ws.send_bytes(frame(1, fill=value))
        result = ws.receive_json()
        assert result['is_final'] and result['frame_count'] == 3
        assert result['window_end_ms'] == 120
        assert fake_model == [([10, 20, 30], True)]
        ws.send_json({'type': 'SESSION_END'})
        assert ws.receive_json()['total_segments'] == 1


def test_empty_final_flushes_existing_frames_without_extra_binary(fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws)
        chunk(ws, count=2)
        ws.send_bytes(frame(1, fill=1)); ws.send_bytes(frame(1, fill=2))
        assert not ws.receive_json()['is_final']
        chunk(ws, cid=2, count=0, begin=80, final=True)
        assert ws.receive_json()['is_final']
        assert fake_model == [([1, 2], False), ([1, 2], True)]


def test_session_end_finalizes_remaining_speech_and_counts_segments(fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws)
        ws.send_bytes(frame(1)); ws.receive_json()
        ws.send_json({'type': 'SESSION_END'})
        assert ws.receive_json()['is_final']
        end = ws.receive_json()
        assert end == {'type': 'SESSION_END', 'total_segments': 1, 'total_frames': 1}
        assert len(fake_model) == 2


def test_consecutive_utterances_do_not_share_frames(fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws, final=True)
        ws.send_bytes(frame(1, fill=10)); ws.receive_json()
        chunk(ws, cid=2, uid=2, begin=1000, final=True)
        ws.send_bytes(frame(2, fill=20)); ws.receive_json()
        assert fake_model == [([10], True), ([20], True)]


def test_abort_discards_active_frames_before_new_utterance(fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws)
        ws.send_bytes(frame(1, fill=10)); ws.receive_json()
        ws.send_json({'type': 'UTTERANCE_ABORT', 'utterance_id': 1})
        assert ws.receive_json()['type'] == 'ABORTED'
        chunk(ws, cid=2, uid=3, begin=80, final=True)
        ws.send_bytes(frame(2, fill=20)); ws.receive_json()
        assert fake_model[-1] == ([20], True)


@pytest.mark.parametrize('payload,code', [
    (b'a', 'BAD_HEADER'), (frame(1, count=100), 'BAD_SIZE'),
    (frame(1, count=ROI_BYTES*4), 'BAD_SIZE'), (frame(1, msg_type=2), 'BAD_MSG_TYPE'),
    (frame(2), 'CHUNK_ID'), (frame(1)[:5] + b'abc' + frame(1)[8:], 'BAD_HEADER'),
], ids=['short_header', 'short_payload', 'float_payload', 'bad_type', 'wrong_chunk', 'reserved'])
def test_invalid_binary_rejected_before_inference(payload, code, fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws)
        ws.send_bytes(payload)
        assert ws.receive_json()['code'] == code
        assert not fake_model


def test_binary_without_chunk_is_protocol_error():
    with client.websocket_connect('/ws/stream') as ws:
        ws.send_bytes(frame(1))
        assert ws.receive_json()['code'] == 'PROTOCOL'


@pytest.mark.parametrize('kind', ['SESSION_END', 'CHUNK_START', 'UTTERANCE_ABORT'])
def test_incomplete_chunk_cannot_be_silently_finalized(kind, fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws, count=2, final=True)
        ws.send_bytes(frame(1))
        ws.send_json({'type': kind})
        assert ws.receive_json()['code'] == 'INCOMPLETE_CHUNK'
        assert not fake_model


@pytest.mark.parametrize('change', [
    {'frame_count': -1}, {'frame_count': 12}, {'frame_count': True},
    {'window_start_ms': 1}, {'window_end_ms': 0}, {'is_final': 'true'},
    {'utterance_id': None}, {'chunk_id': 0}, {'frame_count': 0, 'window_end_ms': 0},
])
def test_invalid_control_fields_rejected(change):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws)
        msg = {'type': 'CHUNK_START', 'chunk_id': 1, 'utterance_id': 1, 'frame_count': 1,
               'window_start_ms': 0, 'window_end_ms': 40, 'is_final': False}
        ws.send_json({**msg, **change})
        assert ws.receive_json()['type'] == 'ERROR'


@pytest.mark.parametrize('second', [
    {'chunk_id': 1, 'window_start_ms': 40, 'window_end_ms': 80},
    {'chunk_id': 2, 'window_start_ms': 0, 'window_end_ms': 40},
    {'chunk_id': 2, 'window_start_ms': 80, 'window_end_ms': 120},
    {'chunk_id': 2, 'utterance_id': 2, 'window_start_ms': 40, 'window_end_ms': 80},
])
def test_duplicate_out_of_order_or_missing_timeline_rejected(second):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws)
        ws.send_bytes(frame(1)); ws.receive_json()
        ws.send_json({'type': 'CHUNK_START', 'utterance_id': 1, 'frame_count': 1, 'is_final': False, **second})
        assert ws.receive_json()['type'] == 'ERROR'


def test_server_buffer_cap_is_enforced(fake_model):
    with client.websocket_connect('/ws/stream') as ws:
        start(ws)
        count = 0
        for cid in range(1, 20):
            chunk(ws, cid=cid, count=11, begin=count*40)
            if count + 11 > WS_MAX_UTTERANCE_FRAMES:
                assert ws.receive_json()['code'] == 'BUFFER_LIMIT'
                break
            for _ in range(11): ws.send_bytes(frame(cid))
            ws.receive_json(); count += 11
        assert count == 198
        assert len(fake_model) == 18


@pytest.mark.parametrize('raw', ['bad', '[]', 'null'])
def test_bad_json(raw):
    with client.websocket_connect('/ws/stream') as ws:
        ws.send_text(raw)
        assert ws.receive_json()['code'] == 'BAD_JSON'


def test_old_protocol_rejected():
    with client.websocket_connect('/ws/stream') as ws:
        ws.send_json({'type': 'SESSION_START', 'session_id': 'old', 'client_ts': 1})
        assert ws.receive_json()['code'] == 'PROTOCOL_VERSION'


def test_model_missing_is_reported_at_session_start(monkeypatch):
    def missing(): raise FileNotFoundError('not trained')
    monkeypatch.setattr(handler, '_get_model', missing)
    with client.websocket_connect('/ws/stream') as ws:
        ws.send_json({'type': 'SESSION_START', 'protocol_version': 2, 'session_id': 's', 'client_ts': 1})
        assert ws.receive_json()['code'] == 'MODEL_NOT_READY'


def test_model_failure_cannot_leave_reusable_buffer(monkeypatch):
    def failed(*args): raise RuntimeError('test failure')
    monkeypatch.setattr(handler, '_infer', failed)
    with client.websocket_connect('/ws/stream') as ws:
        start(ws); chunk(ws); ws.send_bytes(frame(1))
        assert ws.receive_json()['code'] == 'INFERENCE_FAILED'
        with pytest.raises(WebSocketDisconnect): ws.receive_json()


def test_origin_allowed():
    with client.websocket_connect('/ws/stream', headers={'origin': CORS_ORIGINS[0]}) as ws:
        start(ws); ws.send_json({'type': 'SESSION_END'})
        assert ws.receive_json()['total_segments'] == 0


def test_origin_rejected():
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws/stream', headers={'origin': 'http://evil.example.com'}): pass


def test_inference_runs_off_event_loop(monkeypatch):
    async def run():
        main_thread = threading.get_ident()
        thread_ids = []
        def infer(frames, final):
            thread_ids.append(threading.get_ident())
            return '값', None
        monkeypatch.setattr(handler, '_infer', infer)
        class Socket:
            async def send_json(self, value): self.result = value
        state = handler.StreamSession(session_id='s', utterance_id=1)
        state.frames = [np.zeros((96, 96, 3), dtype=np.uint8)]
        await handler._predict(Socket(), state, {'chunk_id': 1, 'is_final': False})
        assert thread_ids and thread_ids[0] != main_thread
    asyncio.run(run())
