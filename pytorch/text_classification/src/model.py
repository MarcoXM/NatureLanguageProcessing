import torch.nn.functional as F
import torch.nn as nn
import torch

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, # How many distinct word in your corpus
                         embed_dim,
                            num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        
        self.embedding.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_uniform_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out_2 = self.fc(dropout(embedded))
            else:
                out_2 += self.fc(dropout(embedded))
        out_2 /= len(self.dropouts)
        return out_2