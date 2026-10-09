"""Discover ESC-style audio without trusting the legacy feature cache."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"^(?P<fold>[1-5])-(?P<source>\d+)-[A-Z](?:-\d+)?$")


def discover(root):
    root = Path(root).resolve()
    rows, seen = [], set()
    for path in sorted(root.rglob('*.ogg')):
        match = PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f'Unrecognized ESC filename: {path.name}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate audio content: {path}')
        seen.add(digest)
        rows.append(dict(path=path.relative_to(root).as_posix(),
                         category=path.parent.name, fold=int(match['fold']),
                         source=match['source'], sha256=digest))
    if not rows:
        raise ValueError(f'No .ogg audio found under {root}')
    classes = sorted({r['category'] for r in rows})
    for r in rows:
        r['label'] = classes.index(r['category'])
    source_folds = {}
    for r in rows:
        source_folds.setdefault(r['source'], set()).add(r['fold'])
    if any(len(v) > 1 for v in source_folds.values()):
        raise ValueError('A source recording occurs in multiple folds')
    for fold in range(1, 6):
        if {r['label'] for r in rows if r['fold'] == fold} != set(range(len(classes))):
            raise ValueError(f'Fold {fold} does not contain every class')
    return rows, classes


def prepare(root, output, sample_rate=22050, seconds=5.0):
    import librosa
    rows, classes = discover(root)
    vectors, specs = [], []
    size = int(sample_rate * seconds)
    for row in rows:
        y, _ = librosa.load(Path(root) / row['path'], sr=sample_rate, mono=True)
        if not len(y) or not np.isfinite(y).all():
            raise ValueError(f"Empty/nonfinite audio: {row['path']}")
        y = librosa.util.fix_length(y, size=size)
        power = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=1024,
                                               hop_length=512, n_mels=64)
        logmel = librosa.power_to_db(power, ref=1.0).astype(np.float32)
        mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)
        blocks = [logmel, mfcc, librosa.feature.delta(mfcc),
                  librosa.feature.zero_crossing_rate(y),
                  librosa.feature.spectral_centroid(y=y, sr=sample_rate),
                  librosa.feature.spectral_bandwidth(y=y, sr=sample_rate),
                  librosa.feature.spectral_rolloff(y=y, sr=sample_rate),
                  librosa.feature.rms(y=y)]
        vectors.append(np.concatenate([stat(b, axis=1) for b in blocks
                                       for stat in (np.mean, np.std)]))
        specs.append(logmel)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(classes=classes, clips=rows, sample_rate=sample_rate,
                    seconds=seconds, n_fft=1024, hop_length=512, n_mels=64,
                    feature_version=1)
    np.savez_compressed(output, features=np.asarray(vectors, dtype=np.float32),
                        spectrograms=np.asarray(specs),
                        labels=np.array([r['label'] for r in rows]),
                        folds=np.array([r['fold'] for r in rows]),
                        metadata=np.array(json.dumps(metadata)))
    return metadata


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    metadata = json.loads(str(result.pop('metadata')))
    n = len(metadata['clips'])
    for key in ('features', 'spectrograms', 'labels', 'folds'):
        if len(result[key]) != n or not np.isfinite(result[key]).all():
            raise ValueError(f'Invalid cache array: {key}')
    if result['features'].ndim != 2 or result['spectrograms'].ndim != 3:
        raise ValueError('Unexpected feature/spectrogram dimensions')
    expected_y = np.array([r['label'] for r in metadata['clips']])
    expected_f = np.array([r['fold'] for r in metadata['clips']])
    if not np.array_equal(result['labels'], expected_y) or not np.array_equal(result['folds'], expected_f):
        raise ValueError('Cache labels/folds do not match manifest')
    sources = {}
    for row in metadata['clips']:
        sources.setdefault(row['source'], set()).add(row['fold'])
    if any(len(v) != 1 for v in sources.values()):
        raise ValueError('Source leakage across folds')
    for fold in range(1, 6):
        if set(result['labels'][result['folds'] == fold]) != set(range(len(metadata['classes']))):
            raise ValueError(f'Incomplete fold {fold}')
    return result, metadata
