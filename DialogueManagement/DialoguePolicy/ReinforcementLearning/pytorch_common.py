from typing import List

import torch
from torch import nn as nn
from torch.distributions import Categorical, Bernoulli


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

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        features = self.convnet(x)
        features_pooled = self.pooling(features).squeeze(2)
        return features_pooled


class CommonDistribution:
    def __init__(self,intent_probs,slot_sigms):
        self.cd = Categorical(intent_probs)
        self.bd = Bernoulli(slot_sigms)

    def sample(self):
        return self.cd.sample(),self.bd.sample()

    def log_prob(self,intent,slots):
        if len(intent.shape) == 1:  # cause its stupid!
            intent = intent.unsqueeze(0)
        log_prob = torch.sum(
            torch.cat([self.cd.log_prob(intent), self.bd.log_prob(slots)], dim=1)
        )
        return log_prob

def calc_discounted_returns(rewards:List[float], gamma:float):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    return returns
