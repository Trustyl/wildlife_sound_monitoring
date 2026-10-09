# Environmental sound classification

Reproducible experiments comparing acoustic-feature classifiers and a compact
spectrogram CNN. The current verified audio subset contains **200 clips, five
classes, and five filename-defined folds**:

- Person sneeze
- Helicopter
- Chainsaw
- Rooster
- Fire crackling

The original repository describes ESC-10 and includes a 400-row, ten-class
`feat.npy`/`label.npy` cache. Only five classes of source audio are checked in,
and that cache has no filename/source manifest. The new experiments deliberately
re-extract features from the available audio instead of guessing cache provenance.
They discover both the top-level class folders and `audio-data/` recursively.

## Run

Use Python 3.12 from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m esc prepare
python -m esc benchmark --jobs 2
```

For the CNN (CPU is supported):

```bash
python -m pip install -r requirements-cnn.txt
python -m esc cnn --epochs 40 --patience 8
# Controlled augmentation ablation, same seed and folds:
python -m esc cnn --epochs 40 --patience 8 --no-augment --output reports/cnn-no-augment.json
```

Direct dependencies are pinned to the versions used for the recorded experiments.
All commands expose `--help`. Outputs default to `artifacts/audio.npz`,
`reports/baselines.json`, and `reports/cnn.json`. Feature preparation needs no
network access after installation. It fails on unreadable audio, duplicate audio
content, missing classes in a fold, or source recordings crossing folds. The cache
records relative paths and SHA-256 hashes, extraction settings, labels and folds.

## Experiments

**Classical models:** majority-class dummy, regularized logistic regression,
RBF SVM, random forest and Extra Trees. Features combine mean and standard
deviation of 64 log-mel bands, 20 MFCCs and their deltas, zero-crossing rate,
centroid, bandwidth, rolloff and RMS (218 features). Each outer test fold is held
out while hyperparameters are selected by macro-F1 over the four remaining
folds. Scaling lives inside the logistic/SVM pipeline, including inner CV.

**CNN:** three 2-D convolution blocks, batch normalization, global average
pooling, dropout, AdamW and cross-entropy with zero-based labels. The network
learns on a 64-band log-mel time–frequency representation, rather than treating
unrelated averaged features as temporal samples. Time/frequency masking applies
only to training batches. Each outer test fold remains untouched; the next fold
is reserved for validation loss/early stopping and three folds train the model.
Normalization uses training data only. CPU seeds and deterministic algorithms
are set explicitly; cross-version numerical equality is not guaranteed.

**Reports:** out-of-fold accuracy, macro-F1, per-class precision/recall/F1,
confusion matrix, fold scores, fold accuracy sample standard deviation, predictions
aligned with the clip manifest, settings and package versions. CNN reports also
include loss histories. Fold standard deviation is descriptive, not a confidence
interval. Classical models train on four outer training folds after tuning; the
CNN trains on three because one is retained for early stopping, so training data
budgets differ. No model superiority is assumed before running the experiments.

See [measured results](reports/RESULTS.md) for the checked-in runs.

## Data and scope

The filenames follow the original [ESC dataset](https://github.com/karolpiczak/ESC-50)
fold/source convention. Source IDs are checked for fold separation to prevent
segments of one recording crossing the train/test boundary. This repository's
folder labels are retained verbatim; this is a partial, locally verified subset,
not a claim of an official full ESC-10 or ESC-50 benchmark.

Five-second mono clips are resampled to 22,050 Hz and padded/truncated to length.
Do not infer real-world wildlife monitoring performance from this small, curated
subset. Broader field audio, unknown sounds, overlapping events and recording
conditions require separate evaluation. Check the upstream dataset's licensing
and attribution requirements before redistributing or using recordings.

## Historical code

The original notebook and seven top-level Python scripts are retained as historical
experiments. They use inconsistent random splits, old Keras APIs and, in some
cases, missing imports or shifted labels. Use `python -m esc ...` for the supported
workflow. The historical reported accuracies are not comparable with the new
five-class, source-separated evaluation and are not used as measured baselines.
Existing source and license attribution are retained in Git history and LICENSE.

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```

Tests cover inventory, duplicate/source leakage rejection, manifest alignment,
nested out-of-fold evaluation, and CNN shapes/gradient flow. The CNN test skips
when PyTorch is absent; GitHub Actions installs CPU PyTorch and runs it.
