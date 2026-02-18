import math
import random

import torch
import torch.nn as nn

from models.bert import BERT
from models.classification_head import ClassificationHead


def make_random_perturbation(config):
    """Return a new config with one SA-optimizable parameter randomly perturbed."""
    new_config = config.copy()
    param = random.choice(['learning_rate', 'dropout', 'n_layers'])

    if param == 'learning_rate':
        factor = random.uniform(0.5, 2.0)
        new_config['learning_rate'] = config['learning_rate'] * factor
    elif param == 'dropout':
        delta = random.uniform(-0.1, 0.1)
        new_config['dropout'] = max(0.0, min(0.5, config['dropout'] + delta))
    elif param == 'n_layers':
        delta = random.randint(-2, 2)
        new_config['n_layers'] = max(1, min(16, config['n_layers'] + delta))

    return new_config


def evaluate_bert_model(config, train_loader, device, n_eval_batches=50):
    """
    Train a fresh BERT model for n_eval_batches steps and return the average loss.
    Lower loss → better config (SA minimises cost).
    """
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

    bert.train()
    head.train()

    total_loss = 0.0
    n_batches = 0
    for seq, seg, labels in train_loader:
        if n_batches >= n_eval_batches:
            break
        seq, seg, labels = seq.to(device), seg.to(device), labels.to(device)
        optimizer.zero_grad()
        out = bert(seq, seg)
        cls_out = out[:, 0, :]          # [CLS] token representation
        logits = head(cls_out)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def simulated_annealing(cost_function, initial_config, temperature, cooling_rate, max_iterations, **kwargs):
    """
    Minimise cost_function over the SA-optimizable hyperparameters.

    Extra keyword arguments are forwarded to cost_function on every call,
    e.g. train_loader=..., device=..., n_eval_batches=...
    """
    current_config = initial_config.copy()
    current_cost = cost_function(current_config, **kwargs)
    best_config = current_config.copy()
    best_cost = current_cost

    for iteration in range(max_iterations):
        if temperature <= 0.01:
            break

        new_config = make_random_perturbation(current_config)
        new_cost = cost_function(new_config, **kwargs)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            current_config = new_config
            current_cost = new_cost

            if current_cost < best_cost:
                best_config = current_config.copy()
                best_cost = current_cost

        temperature *= cooling_rate
        print(
            f"SA iter {iteration + 1:3d}: cost={current_cost:.4f}  "
            f"best={best_cost:.4f}  temp={temperature:.4f}"
        )

    return best_config
