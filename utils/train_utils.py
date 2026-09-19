import torch


def compute_accuracy(bert, head, loader, device, max_batches=None):
    """Classification accuracy of the [CLS] head over a dataloader."""
    bert.eval()
    head.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (seq, seg, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            seq, seg, labels = seq.to(device), seg.to(device), labels.to(device)
            out = bert(seq, seg)
            logits = head(out[:, 0, :])
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)
