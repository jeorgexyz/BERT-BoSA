from models.bert import BERT
from models.classification_head import ClassificationHead

# Instantiate the BERT model
bert_model = BERT(vocab_size, max_len, embedding_dim, n_segments, n_layers, attn_heads, dropout)

# Instantiate the classification head
classification_head = ClassificationHead(bert_hidden_size=EMBEDDING_DIM, num_classes=4)  # 4 classes for AG_NEWS