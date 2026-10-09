"""Compact 2-D log-mel CNN with training-only augmentation and early stopping."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .benchmark import metrics, save_report
from .data import load_cache


class SoundCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, classes))

    def forward(self, x):
        return self.network(x)


def mask_spectrograms(batch):
    """Mask standardized values with zero (the training mean)."""
    batch = batch.clone()
    for x in batch:
        for axis, maximum in ((1, 8), (2, 24)):
            width = int(torch.randint(0, min(maximum, x.shape[axis]) + 1, (1,)))
            start = int(torch.randint(0, x.shape[axis] - width + 1, (1,)))
            if axis == 1:
                x[:, start:start + width, :] = 0
            else:
                x[:, :, start:start + width] = 0
    return batch


def run(cache, output, seed=42, epochs=40, patience=8, augment=True, threads=2):
    if epochs < 1 or patience < 1 or threads < 1:
        raise ValueError('epochs, patience and threads must be positive')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data, metadata = load_cache(cache)
    X, y, folds = data['spectrograms'], data['labels'], data['folds']
    predictions = np.full(len(y), -1, dtype=int)
    scores = []
    for fold in range(1, 6):
        random.seed(seed + fold)
        np.random.seed(seed + fold)
        torch.manual_seed(seed + fold)
        validation_fold = fold % 5 + 1
        train = (folds != fold) & (folds != validation_fold)
        val, test = folds == validation_fold, folds == fold
        mean, std = float(X[train].mean()), max(float(X[train].std()), 1e-6)
        tensors = torch.from_numpy(((X - mean) / std).astype(np.float32))[:, None]
        labels = torch.from_numpy(y.astype(np.int64))
        loader = DataLoader(TensorDataset(tensors[train], labels[train]), batch_size=32,
                            shuffle=True, generator=torch.Generator().manual_seed(seed + fold))
        model = SoundCNN(len(metadata['classes']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch, target in loader:
                optimizer.zero_grad()
                loss = criterion(model(mask_spectrograms(batch) if augment else batch), target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(target)
            model.eval()
            with torch.no_grad():
                validation_loss = float(criterion(model(tensors[val]), labels[val]))
            history.append(dict(epoch=epoch, train_loss=total_loss / int(train.sum()), validation_loss=validation_loss))
            if validation_loss < best_loss:
                best_loss, best_state, best_epoch, stale = validation_loss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= patience:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions[test] = model(tensors[test]).argmax(1).numpy()
        score = metrics(y[test], predictions[test], metadata['classes'])
        scores.append(dict(fold=fold, validation_fold=validation_fold, best_epoch=best_epoch,
                           accuracy=score['accuracy'], macro_f1=score['macro_f1'], history=history))
        print(f"CNN fold {fold}: accuracy={score['accuracy']:.3f}, best epoch={best_epoch}", flush=True)
    result = dict(**metrics(y, predictions, metadata['classes']), folds=scores,
                  predictions=predictions.tolist(), augmentation=augment, max_epochs=epochs,
                  patience=patience, fold_accuracy_std=float(np.std([s['accuracy'] for s in scores], ddof=1)))
    return save_report(output, metadata, {'cnn': result}, seed,
                       'Five outer folds; next fold used only for early stopping; three training folds. CPU deterministic training, training-only scalar normalization and SpecAugment.')
