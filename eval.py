import argparse

import torch

from models.bert import BERT
from models.classification_head import ClassificationHead
from utils.data_utils import Vocab, get_dataloaders
from utils.train_utils import compute_accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained BERT-BoSA checkpoint')
    parser.add_argument('--checkpoint', default='model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    vocab = Vocab(checkpoint['vocab'])  # exact vocab from training, never rebuilt

    print("Loading data...")
    _, _, test_loader, _ = get_dataloaders(config, vocab=vocab)

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

    accuracy = compute_accuracy(bert, head, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
