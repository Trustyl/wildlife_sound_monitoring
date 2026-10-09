import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from esc.benchmark import models, run
from esc.data import discover, load_cache


def test_real_audio_inventory():
    rows, classes = discover(Path(__file__).resolve().parents[1])
    assert len(rows) >= 200
    assert len(classes) >= 5
    assert len({r['sha256'] for r in rows}) == len(rows)
    for fold in range(1, 6):
        assert {r['category'] for r in rows if r['fold'] == fold} == set(classes)


def test_source_leakage_rejected(tmp_path):
    for fold in range(1, 6):
        d = tmp_path / 'class'
        d.mkdir(exist_ok=True)
        (d / f'{fold}-123-A.ogg').write_bytes(bytes([fold]))
    with pytest.raises(ValueError, match='multiple folds'):
        discover(tmp_path)


def test_duplicate_audio_rejected(tmp_path):
    for fold in (1, 2):
        (tmp_path / f'{fold}-{fold}-A.ogg').write_bytes(b'identical')
    with pytest.raises(ValueError, match='Duplicate'):
        discover(tmp_path)


def make_cache(tmp_path):
    y = np.tile([0, 1, 0, 1], 5)
    folds = np.repeat(np.arange(1, 6), 4)
    rows = [dict(label=int(a), fold=int(b), source=str(i)) for i, (a, b) in enumerate(zip(y, folds))]
    path = tmp_path / 'data.npz'
    np.savez(path, features=np.column_stack([y, y + 1]).astype(float),
             spectrograms=np.zeros((20, 64, 32)), labels=y, folds=folds,
             metadata=np.array(json.dumps(dict(classes=['a', 'b'], clips=rows))))
    return path


def test_nested_oof_and_metrics(tmp_path):
    path = make_cache(tmp_path)
    report = run(path, tmp_path / 'result.json', ['dummy', 'logistic'])
    assert report['results']['logistic']['accuracy'] == 1.0
    assert report['results']['dummy']['accuracy'] == 0.5
    assert len(report['results']['logistic']['predictions']) == 20
    assert len(report['results']['logistic']['folds']) == 5
    assert isinstance(models(42)['svm'][0], Pipeline)


def test_cache_metadata_alignment(tmp_path):
    path = make_cache(tmp_path)
    with np.load(path) as d:
        arrays = dict(d)
    arrays['labels'] = 1 - arrays['labels']
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='manifest'):
        load_cache(path)


def test_cnn_shape_and_augmentation():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    from esc.cnn import SoundCNN, mask_spectrograms
    x = torch.ones(3, 1, 64, 48)
    result = mask_spectrograms(x)
    assert torch.all(x == 1)  # augmentation must not modify the cache
    assert result.shape == x.shape
    logits = SoundCNN(5)(result)
    assert logits.shape == (3, 5)
    logits.sum().backward()
