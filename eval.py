import torch

from models.bert import BERT
from models.classification_head import ClassificationHead
from utils.data_utils import get_dataloaders


def evaluate(bert, head, loader, device):
    bert.eval()
    head.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for seq, seg, labels in loader:
            seq, seg, labels = seq.to(device), seg.to(device), labels.to(device)
            out = bert(seq, seg)
            cls_out = out[:, 0, :]      # [CLS] token
            logits = head(cls_out)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('model.pt', map_location=device)
    config = checkpoint['config']

    print("Loading data...")
    _, test_loader, _ = get_dataloaders(config)

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

    bert.load_state_dict(checkpoint['bert'])
    head.load_state_dict(checkpoint['head'])

    accuracy = evaluate(bert, head, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
