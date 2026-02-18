import torch
import torch.nn as nn

from config import FIXED_CONFIG, initial_config
from models.bert import BERT
from models.classification_head import ClassificationHead
from optimization.simulated_annealing import evaluate_bert_model, simulated_annealing
from utils.data_utils import get_dataloaders


def train_epoch(bert, head, loader, optimizer, criterion, device):
    bert.train()
    head.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seq, seg, labels in loader:
        seq, seg, labels = seq.to(device), seg.to(device), labels.to(device)
        optimizer.zero_grad()
        out = bert(seq, seg)
        cls_out = out[:, 0, :]      # [CLS] token
        logits = head(cls_out)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Merge SA-optimizable params with fixed hyperparameters
    config = {**FIXED_CONFIG, **initial_config}

    print("Loading data...")
    train_loader, test_loader, vocab = get_dataloaders(config)
    config['vocab_size'] = len(vocab)
    print(f"Vocab size: {config['vocab_size']}")

    # Run Simulated Annealing to find the best hyperparameter config
    print("Running Simulated Annealing to optimise configuration...")
    best_config = simulated_annealing(
        evaluate_bert_model,
        config,
        temperature=1.0,
        cooling_rate=0.9,
        max_iterations=5,       # increase for a more thorough search
        train_loader=train_loader,
        device=device,
        n_eval_batches=20,
    )
    print(f"\nBest config: {best_config}\n")

    # Full training with the best config
    print("Training with best config...")
    bert = BERT(
        best_config['vocab_size'],
        best_config['max_len'],
        best_config['embedding_dim'],
        best_config['n_segments'],
        best_config['n_layers'],
        best_config['attn_heads'],
        best_config['dropout'],
    ).to(device)
    head = ClassificationHead(best_config['embedding_dim'], best_config['num_classes']).to(device)

    optimizer = torch.optim.Adam(
        list(bert.parameters()) + list(head.parameters()),
        lr=best_config['learning_rate'],
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(best_config['epochs']):
        loss, acc = train_epoch(bert, head, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{best_config['epochs']}  loss={loss:.4f}  acc={acc:.4f}")

    torch.save(
        {'bert': bert.state_dict(), 'head': head.state_dict(), 'config': best_config},
        'model.pt',
    )
    print("Saved model to model.pt")


if __name__ == '__main__':
    main()
