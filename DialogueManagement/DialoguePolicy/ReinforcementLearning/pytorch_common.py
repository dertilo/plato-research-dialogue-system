from typing import List, Tuple, Dict, NamedTuple

import torch
from torch import nn as nn
from torch.distributions import Categorical, Bernoulli
from torch.nn import functional as F
from torchtext.data import Example

from Dialogue.Action import DialogueAct
from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.dialogue_common import state_to_json


class StateEncoder(nn.Module):
    def __init__(
        self, vocab_size, encode_dim=64, embed_dim=32, padding_idx=None
    ) -> None:
        super().__init__()
        hidden_dim = encode_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
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
    def __init__(self, intent_probs, slot_sigms):
        self.cd = Categorical(intent_probs)
        self.bd = Bernoulli(slot_sigms)

    def sample(self):
        return self.cd.sample(), self.bd.sample()

    def log_prob(self, intent, slots):
        intent = intent.squeeze()
        cd_log_prob = self.cd.log_prob(intent).unsqueeze(1)
        bd_log_prob = self.bd.log_prob(slots)
        log_prob = torch.sum(torch.cat([cd_log_prob, bd_log_prob], dim=1), dim=1,)
        return log_prob

    def entropy(self):
        bd_entr = self.bd.entropy().mean(dim=1)
        cd_entr = self.cd.entropy()
        return bd_entr + cd_entr


def calc_discounted_returns(rewards: List[float], gamma: float):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def tokenize(text_field, state: SlotFillingDialogueState):
    state_string = state_to_json(state)
    example = Example.fromlist([state_string], [("dialog_state", text_field)])
    tokens = [t for t in example.dialog_state if t in text_field.vocab.stoi]
    return tokens


class DialogTurn(NamedTuple):
    act: DialogueAct
    tokens: List[str]
    reward: float
    returnn: float


def process_dialogue_to_turns(
    text_field, dialogue: List[Dict], gamma=0.99
) -> List[DialogTurn]:
    x = [(d["action"], d["state"], d["reward"]) for d in dialogue]

    rewards = [t["reward"] for t in dialogue]
    returns = calc_discounted_returns(rewards, gamma)
    turns = [
        DialogTurn(a[0], tokenize(text_field, s), r, ret)
        for (a, s, r), ret in zip(x, returns)
    ]
    return turns


class Actor(nn.Module):
    def __init__(self, hidden_dim, num_intents, num_slots) -> None:
        super().__init__()
        self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.slots_head = nn.Linear(hidden_dim, num_slots)

    def forward(self, x):
        intent_probs = F.softmax(self.intent_head(x), dim=1)
        slots_sigms = torch.sigmoid(self.slots_head(x))
        return intent_probs, slots_sigms