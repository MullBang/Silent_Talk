import copy
import pytest
import execution as e


def fixture():
    tasks = [{'kind': 'phrase', 'lr': .001, 'clip_ids': ['a', 'b']},
             {'kind': 'phrase', 'lr': .0003, 'clip_ids': ['c', 'd']}]
    state = {'model': {'w': e.torch.ones(2)},
             'optimizer': {'state': {0: {'step': e.torch.tensor(1.)}}, 'param_groups': [{'lr': .001}]},
             'rng': {'cpu': e.torch.arange(3)}}
    identity = {'mode': 'preflight', 'seed': 42, 'arm': 'P_phrase'}
    payload = {'schema': 'offline-stage5c-complete-checkpoint-v1', 'identity': identity,
               'state': state, 'state_hashes': e.old.hashes(state), 'completed_updates': 1,
               'history': [{'update': 1, 'kind': 'phrase', 'learning_rate': .001,
                            'clip_ids': ['a', 'b'], 'optimizer_updated': True}],
               'next_task': tasks[1], 'evaluation_due': False}
    return payload, identity, tasks


def test_valid_resume():
    e.validate_checkpoint(*fixture())


@pytest.mark.parametrize('fault', ['identity', 'cursor', 'weights', 'adam', 'lr', 'input', 'kind', 'pending'])
def test_corrupt_resume_rejected(fault):
    p, identity, tasks = fixture(); p = copy.deepcopy(p)
    if fault == 'identity': p['identity']['seed'] = 43
    elif fault == 'cursor': p['next_task'] = tasks[0]
    elif fault == 'weights': p['state']['model']['w'][0] = 99
    elif fault == 'input': p['history'][0]['clip_ids'].reverse()
    elif fault == 'kind': p['history'][0]['kind'] = 'full_sentence'
    elif fault == 'pending': p['evaluation_due'] = True
    else:
        if fault == 'adam': p['state']['optimizer']['state'][0]['step'] += 1
        else: p['state']['optimizer']['param_groups'][0]['lr'] = .0003
        p['state_hashes'] = e.old.hashes(p['state'])
    with pytest.raises(ValueError): e.validate_checkpoint(p, identity, tasks)


@pytest.mark.parametrize('fault', ['checkpoint', 'state', 'failed'])
def test_failed_evaluation_cannot_authorize_more_training(fault, monkeypatch):
    monkeypatch.setattr(e, 'check', lambda value: None)
    checkpoint, state = {'path': 'a', 'sha256': 'x'}, {'model': 'a', 'optimizer': 'b', 'rng': 'c'}
    receipt = {'status': 'completed', 'checkpoint': checkpoint.copy(),
               'training_state_before': state.copy(), 'training_state_after': state.copy(),
               'evaluation': {'summary': {}, 'failures': 0}}
    if fault == 'checkpoint': receipt['checkpoint']['sha256'] = 'y'
    elif fault == 'state': receipt['training_state_after']['rng'] = 'd'
    else: receipt['evaluation']['failures'] = 1
    with pytest.raises(ValueError): e.assert_receipt(receipt, checkpoint, state)


@pytest.mark.parametrize('seed', [42, 43])
def test_real_phrase_only_schedule_and_lr_seams(seed):
    source = e.S5B / f'design/proposed_phrase_schedule_seed{seed}.json'
    tasks = e.tasks_for({'schedules': [{'seed': seed, **e.ref(source)}]}, seed, 'main')
    assert len(tasks) == 1680 and all(row['kind'] == 'phrase' for row in tasks)
    assert tasks[419]['kind'] == tasks[420]['kind'] == 'phrase'
    assert tasks[839]['lr'] == .001 and tasks[840]['lr'] == .0003
    assert tasks[1049]['lr'] == .0003 and tasks[1050]['lr'] == tasks[-1]['lr'] == .0001
    assert sum(sum(row['lengths']) for row in tasks) == 134680
    assert tasks[1260]['clip_ids'] == tasks[0]['clip_ids']
