initial_config = {
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'n_layers': 8,
}

FIXED_CONFIG = {
    'vocab_size': 30000,    # updated after vocab is built from data
    'max_len': 256,
    'embedding_dim': 512,
    'n_segments': 2,
    'attn_heads': 8,
    'batch_size': 32,
    'epochs': 3,
    'num_classes': 4,       # AG_NEWS has 4 classes
}
