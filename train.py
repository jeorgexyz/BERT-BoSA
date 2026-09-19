import argparse
import csv
import random

import torch
import torch.nn as nn

from config import FIXED_CONFIG, initial_config
from models.bert import BERT
from models.classification_head import ClassificationHead
from optimization.simulated_annealing import evaluate_bert_model, simulated_annealing
from utils.data_utils import get_dataloaders
from utils.train_utils import compute_accuracy


def train_epoch(bert, head, loader, optimizer, criterion, device):
    bert.train()
    head.train()
    params = list(bert.parameters()) + list(head.parameters())
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def parse_args():
    parser = argparse.ArgumentParser(description='Train BERT-BoSA on AG_NEWS')
    parser.add_argument('--epochs', type=int, default=FIXED_CONFIG['epochs'],
                        help='epochs for the final full training run')
    parser.add_argument('--sa-iterations', type=int, default=15,
                        help='simulated annealing iterations')
    parser.add_argument('--sa-train-batches', type=int, default=100,
                        help='training batches per SA candidate evaluation')
    parser.add_argument('--skip-sa', action='store_true',
                        help='skip the SA search and train with the initial config')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='model.pt', help='checkpoint path')
    parser.add_argument('--sa-log', default='sa_log.csv',
                        help='CSV file for the per-iteration SA trace (see plot_sa.py)')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Merge SA-optimizable params with fixed hyperparameters
    config = {**FIXED_CONFIG, **initial_config}
    config['epochs'] = args.epochs

    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(config, seed=args.seed)
    config['vocab_size'] = len(vocab)
    print(f"Vocab size: {config['vocab_size']}")

    if args.skip_sa:
        best_config = config
    else:
        # Run Simulated Annealing to find the best hyperparameter config.
        # Cost is 1 - val_accuracy, so temperature is on the accuracy scale:
        # 0.1 initially accepts ~5-point accuracy regressions with p≈0.6.
        # The cooling rate is derived so the temperature decays from 0.1 to
        # 0.001 over however many iterations were requested.
        start_temp, final_temp = 0.1, 0.001
        cooling_rate = (final_temp / start_temp) ** (1 / max(args.sa_iterations, 1))
        print("Running Simulated Annealing to optimise configuration...")
        best_config, history = simulated_annealing(
            evaluate_bert_model,
            config,
            temperature=start_temp,
            cooling_rate=cooling_rate,
            max_iterations=args.sa_iterations,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_train_batches=args.sa_train_batches,
            seed=args.seed,
        )
        print(f"\nBest config: {best_config}\n")

        with open(args.sa_log, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved SA trace to {args.sa_log}")

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
        val_acc = compute_accuracy(bert, head, val_loader, device)
        print(f"Epoch {epoch + 1}/{best_config['epochs']}  "
              f"loss={loss:.4f}  train_acc={acc:.4f}  val_acc={val_acc:.4f}")

    torch.save(
        {
            'bert': bert.state_dict(),
            'head': head.state_dict(),
            'config': best_config,
            'vocab': vocab.token2idx,   # eval must use the exact training vocab
        },
        args.output,
    )
    print(f"Saved model to {args.output}")


if __name__ == '__main__':
    main()
