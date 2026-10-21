"""Negative identity/cursor/state tests; no model or optimizer is instantiated."""
import copy
import importlib.util
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probe_under_test', HERE / 'run_probe.py')
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


@pytest.fixture
def payload(monkeypatch):
    """Provide a valid final-cursor shape while replacing only expensive hashing."""
    monkeypatch.setattr(probe.old, 'hashes', lambda state: {'all': 'fixed'})
    return {'schema': 'offline-stage5c-complete-checkpoint-v1', 'completed_updates': 1680,
            'identity': {'seed': 42, 'mode': 'main', 'arm': 'P_phrase', 'runtime': {'fixed': True}},
            'state_hashes': {'all': 'fixed'}, 'next_task': None,
            'state': {'optimizer': {'state': {0: {'step': 1680}}}},
            'history': [{'update': i, 'kind': 'phrase', 'optimizer_updated': True} for i in range(1, 1681)]}


def test_valid_final(payload):
    """Accept the fixed final identity without a neural call."""
    probe.validate_payload(payload, 42, {'fixed': True})


@pytest.mark.parametrize('field,value', [('schema', 'old'), ('completed_updates', 1260),
                                      ('completed_updates', 1680.0), ('next_task', {'update': 1681}),
                                      ('state_hashes', {'all': 'changed'})])
def test_reject_bad_cursor_or_state(payload, field, value):
    """Reject non-final or altered full-state evidence."""
    payload[field] = value
    with pytest.raises(RuntimeError):
        probe.validate_payload(payload, 42, {'fixed': True})


@pytest.mark.parametrize('field,value', [('seed', 43), ('mode', 'preflight'), ('arm', 'A_sentence'), ('runtime', {})])
def test_reject_changed_model_identity(payload, field, value):
    """Reject a different seed/run or numerical runtime."""
    payload['identity'][field] = value
    with pytest.raises(RuntimeError):
        probe.validate_payload(payload, 42, {'fixed': True})


def test_reject_changed_training_history(payload):
    """A sentence-trained history cannot masquerade as the phrase-only model."""
    payload['history'][0]['kind'] = 'full_sentence'
    with pytest.raises(RuntimeError):
        probe.validate_payload(payload, 42, {'fixed': True})


def test_reject_wrong_adam_history(payload):
    """Read-only verification still enforces historical optimizer cursor."""
    payload['state']['optimizer']['state'][0]['step'] = 1260
    with pytest.raises(RuntimeError):
        probe.validate_payload(payload, 42, {'fixed': True})
