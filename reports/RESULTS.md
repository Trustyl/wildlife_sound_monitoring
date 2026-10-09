# Recorded experiments

Run on the 200 checked-in clips (40 per class), five filename-defined folds, seed 42.
Every clip has exactly one outer held-out prediction. These results describe the five-class subset only.

| Model | Accuracy | Macro-F1 | Fold accuracy SD |
|---|---:|---:|---:|
| dummy | 20.0% | 0.0667 | 0.0% |
| logistic | 92.5% | 0.9252 | 4.0% |
| svm | 91.0% | 0.9102 | 4.5% |
| forest | 86.5% | 0.8641 | 2.2% |
| extra_trees | 88.0% | 0.8796 | 2.1% |
| CNN + masking | 84.0% | 0.8379 | 3.8% |
| CNN without masking | 88.0% | 0.8791 | 4.8% |

## Interpretation

Logistic regression is the strongest measured model in this run. Its 15 errors include 11 helicopter/chainsaw confusions. Rooster classification is correct for all 40 held-out clips. This is descriptive performance on a small curated subset, not a deployment estimate.

The CNN uses three training folds plus one early-stopping fold; classical models use four training folds after nested tuning. Both CNN runs use the same fold assignment, seed, optimizer, 40-epoch maximum and patience of eight. The augmentation comparison changes only training-time time/frequency masking. Masking did not improve aggregate performance in this run; it should not be assumed beneficial on this small dataset.

No outer-fold results were used to change these model configurations during this run. Comparing many future runs against the same folds can still overfit the benchmark; reserve new recordings for a final evaluation before deployment.

## Reproduce

```bash
python -m esc prepare
python -m esc benchmark --jobs 2
python -m esc cnn --epochs 40 --patience 8
python -m esc cnn --epochs 40 --patience 8 --no-augment --output reports/cnn-no-augment.json
```

Full fold metrics, confusion matrices, per-class scores, clip hashes and versions:

- [Classical models](baselines.json)
- [CNN with masking](cnn.json)
- [CNN without masking](cnn-no-augment.json)

## Next experiments

Collect the missing five classes with verified source metadata; evaluate additional independent field recordings; investigate helicopter/chainsaw errors with longer temporal features or pretrained audio embeddings. Compare multiple predetermined seeds before drawing conclusions about augmentation. These are future experiments, not completed results.
