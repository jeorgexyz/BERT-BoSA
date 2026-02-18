import torch
import torch.nn as nn


class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_segments, dropout, max_len):
        super(BERTEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_len, embedding_dim)
        self.segment_embeddings = nn.Embedding(n_segments, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # register_buffer ensures this moves to the correct device with .to(device)
        self.register_buffer('position_inp', torch.arange(0, max_len).unsqueeze(0))

    def forward(self, seq, seg):
        seq_len = seq.size(1)
        embed_val = (
            self.token_embeddings(seq)
            + self.segment_embeddings(seg)
            + self.position_embeddings(self.position_inp[:, :seq_len])
        )
        embed_val = self.dropout(embed_val)
        return embed_val


class BERT(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim, n_segments, n_layers, attn_heads, dropout):
        super(BERT, self).__init__()
        self.embeddings = BERTEmbeddings(vocab_size, embedding_dim, n_segments, dropout, max_len)
        # batch_first=True so input/output shape is (batch, seq_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead=attn_heads, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

    def forward(self, seq, seg):
        out = self.embeddings(seq, seg)
        out = self.encoder_block(out)
        return out


if __name__ == "__main__":
    VOCAB_SIZE = 30000
    N_SEGMENTS = 2
    MAX_LENGTH = 256
    EMBEDDING_DIM = 512
    N_LAYERS = 8
    ATTN_HEADS = 8
    DROPOUT = 0.2

    # Use a batch of 2 sequences — shape (batch, seq_len)
    sample_seq = torch.randint(high=VOCAB_SIZE, size=(2, MAX_LENGTH))
    sample_seg = torch.randint(high=N_SEGMENTS, size=(2, MAX_LENGTH))

    embedding = BERTEmbeddings(VOCAB_SIZE, EMBEDDING_DIM, N_SEGMENTS, DROPOUT, MAX_LENGTH)
    embedding_tensor = embedding(sample_seq, sample_seg)
    print(f"Embedding tensor size: {embedding_tensor.size()}")

    bert = BERT(VOCAB_SIZE, MAX_LENGTH, EMBEDDING_DIM, N_SEGMENTS, N_LAYERS, ATTN_HEADS, DROPOUT)
    out = bert(sample_seq, sample_seg)
    print(f"Output tensor size: {out.size()}")
