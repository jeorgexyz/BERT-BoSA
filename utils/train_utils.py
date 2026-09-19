import torch
import torch.nn as nn


def evaluate_model(bert, head, loader, device, max_batches=None):
    """
    Mean cross-entropy loss and accuracy of the [CLS] head over a dataloader.

    Loss is reported alongside accuracy because accuracy is too coarse to
    compare weakly-trained models: after a short training budget every
    candidate still predicts one class and scores exactly chance, while their
    losses already differ.
    """
    bert.eval()
    head.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (seq, seg, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            seq, seg, labels = seq.to(device), seg.to(device), labels.to(device)
            out = bert(seq, seg)
            logits = head(out[:, 0, :])
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    denom = max(total, 1)
    return total_loss / denom, correct / denom


def compute_accuracy(bert, head, loader, device, max_batches=None):
    """Classification accuracy of the [CLS] head over a dataloader."""
    _, accuracy = evaluate_model(bert, head, loader, device, max_batches=max_batches)
    return accuracy
