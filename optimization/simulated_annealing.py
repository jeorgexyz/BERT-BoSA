import math
import random

import torch
import torch.nn as nn

from models.bert import BERT
from models.classification_head import ClassificationHead
from utils.train_utils import compute_accuracy


def make_random_perturbation(config):
    """Return a new config with one SA-optimizable parameter randomly perturbed."""
    new_config = config.copy()
    param = random.choice(['learning_rate', 'dropout', 'n_layers'])

    if param == 'learning_rate':
        factor = random.uniform(0.5, 2.0)
        # Clamp so repeated multiplicative steps can't drift to absurd values
        new_config['learning_rate'] = max(1e-5, min(1e-3, config['learning_rate'] * factor))
    elif param == 'dropout':
        delta = random.uniform(-0.1, 0.1)
        new_config['dropout'] = max(0.0, min(0.5, config['dropout'] + delta))
    elif param == 'n_layers':
        delta = random.randint(-2, 2)
        new_config['n_layers'] = max(1, min(16, config['n_layers'] + delta))

    return new_config


def evaluate_bert_model(config, train_loader, val_loader, device,
                        n_train_batches=100, n_val_batches=50, seed=42):
    """
    Train a fresh BERT model for n_train_batches steps, then score it on
    held-out validation batches. Returns 1 - val_accuracy (SA minimises cost).

    The seed is fixed so every candidate config sees the same initialisation
    and batch order — otherwise run-to-run noise swamps the differences
    between configs and SA accepts/rejects on coin flips.
    """
    torch.manual_seed(seed)

    bert = BERT(
        config['vocab_size'],
        config['max_len'],
        config['embedding_dim'],
        config['n_segments'],
        config['n_layers'],
        config['attn_heads'],
        config['dropout'],
    ).to(device)
    head = ClassificationHead(config['embedding_dim'], config['num_classes']).to(device)

    optimizer = torch.optim.Adam(
        list(bert.parameters()) + list(head.parameters()),
        lr=config['learning_rate'],
    )
    criterion = nn.CrossEntropyLoss()

    params = list(bert.parameters()) + list(head.parameters())

    bert.train()
    head.train()

    # Re-seed after building the model: configs with different n_layers
    # consume different amounts of RNG during init, and the DataLoader draws
    # its shuffle seed from the global RNG when iteration starts.
    torch.manual_seed(seed)

    n_batches = 0
    for seq, seg, labels in train_loader:
        if n_batches >= n_train_batches:
            break
        seq, seg, labels = seq.to(device), seg.to(device), labels.to(device)
        optimizer.zero_grad()
        out = bert(seq, seg)
        cls_out = out[:, 0, :]          # [CLS] token representation
        logits = head(cls_out)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        n_batches += 1

    val_acc = compute_accuracy(bert, head, val_loader, device, max_batches=n_val_batches)
    return 1.0 - val_acc


def simulated_annealing(cost_function, initial_config, temperature, cooling_rate, max_iterations, **kwargs):
    """
    Minimise cost_function over the SA-optimizable hyperparameters.

    Extra keyword arguments are forwarded to cost_function on every call,
    e.g. train_loader=..., device=..., n_train_batches=...

    Returns (best_config, history), where history is a list of dicts, one per
    evaluated candidate.
    """
    current_config = initial_config.copy()
    current_cost = cost_function(current_config, **kwargs)
    best_config = current_config.copy()
    best_cost = current_cost

    # One row per evaluated candidate (iteration 0 is the initial config),
    # so the search trajectory can be saved and plotted afterwards.
    history = [_history_row(0, temperature, current_config, current_cost,
                            True, current_cost, best_cost)]

    for iteration in range(max_iterations):
        new_config = make_random_perturbation(current_config)
        new_cost = cost_function(new_config, **kwargs)
        delta_cost = new_cost - current_cost

        # Temperature used for this acceptance decision, before cooling
        step_temp = temperature
        accepted = delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature)
        if accepted:
            current_config = new_config
            current_cost = new_cost

            if current_cost < best_cost:
                best_config = current_config.copy()
                best_cost = current_cost

        history.append(_history_row(iteration + 1, step_temp, new_config, new_cost,
                                    accepted, current_cost, best_cost))
        temperature *= cooling_rate
        print(
            f"SA iter {iteration + 1:3d}: candidate={new_cost:.4f} "
            f"({'accepted' if accepted else 'rejected'})  cost={current_cost:.4f}  "
            f"best={best_cost:.4f}  temp={step_temp:.4f}"
        )

    return best_config, history


def _history_row(iteration, temperature, config, cost, accepted, current_cost, best_cost):
    return {
        'iteration': iteration,
        'temperature': temperature,
        'learning_rate': config['learning_rate'],
        'dropout': config['dropout'],
        'n_layers': config['n_layers'],
        'candidate_cost': cost,
        'accepted': accepted,
        'current_cost': current_cost,
        'best_cost': best_cost,
    }
