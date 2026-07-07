"""Post-hoc per-class temperature and Platt scaling: fit on validation logits, apply to any logits."""

import numpy as np
import torch

from model import NUM_CLASSES


def fit_temperature(val_logits, val_labels, max_iter=100):
    """Fit one temperature scalar per class by minimizing BCEWithLogitsLoss on validation logits.

    Optimizes log(T) via LBFGS so T = exp(log_T) stays positive. Because each class's
    sigmoid output is independent, the 3 temperatures decouple and can be fit jointly
    in a single LBFGS run (equivalent to fitting each in isolation).
    """
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.float32)

    log_temperature = torch.nn.Parameter(torch.zeros(NUM_CLASSES))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return torch.exp(log_temperature).detach().numpy()


def apply_temperature(logits, temperature):
    """Divide logits by their per-class temperature. Monotonic per class, so AUC ranking is unchanged."""
    return logits / temperature


def fit_platt(val_logits, val_labels, max_iter=100):
    """Fit per-class scale a and shift b (a * logit + b) minimizing BCEWithLogitsLoss on validation logits.

    Unlike temperature scaling (a symmetric rescale toward 0.5), Platt scaling can also
    shift the whole probability distribution, which is the fix for systematic over/under-
    confidence rather than just over-sharp/over-flat confidence. Fit in logit space, on
    validation only. The 3 classes decouple, so fit jointly in one LBFGS run.
    """
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.float32)

    a = torch.nn.Parameter(torch.ones(NUM_CLASSES))
    b = torch.nn.Parameter(torch.zeros(NUM_CLASSES))
    optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(a * logits + b, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return a.detach().numpy(), b.detach().numpy()


def apply_platt(logits, a, b):
    """a * logits + b. Still logits; sigmoid applied downstream. AUC-preserving only where a > 0."""
    return a * logits + b


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
