import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, bert_hidden_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(bert_hidden_size, num_classes)

    def forward(self, inputs):
        dropout_output = self.dropout(inputs)
        linear_output = self.linear(dropout_output)
        return linear_output