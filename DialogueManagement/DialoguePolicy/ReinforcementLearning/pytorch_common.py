from torch import nn as nn
from torch.nn.modules.module import T_co


class StateEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, embed_dim=32,
                 ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3),
            nn.ELU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x) -> T_co:
        x = self.embedding(x)
        x = x.transpose(2, 1)
        features = self.convnet(x)
        features_pooled = self.pooling(features).squeeze(2)
        return features_pooled