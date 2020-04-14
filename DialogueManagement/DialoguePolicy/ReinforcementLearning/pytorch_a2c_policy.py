import os
import re
import shutil
from os import chdir
from typing import List, Dict, Tuple

import numpy
import random

from sklearn import preprocessing
from torchtext.data import Field, Example
from torchtext.vocab import Vocab

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli

from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_common import \
    StateEncoder, sample_from_distr
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_reinforce_policy import (
    ActionEncoder,
    PolicyAgent,
    PyTorchReinforcePolicy,
)
from DialogueManagement.DialoguePolicy.dialogue_common import (
    create_random_dialog_act,
    Domain,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValueDialogAct(DialogueAct):

    def __init__(self, intent='', params=None,value:float=0.0):
        super().__init__(intent, params)
        self.value = value

class Actor(nn.Module):

    def __init__(self,hidden_dim,num_intents,num_slots) -> None:
        super().__init__()
        self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.slots_head = nn.Linear(hidden_dim, num_slots)

    def forward(self, x):
        intent_probs = F.softmax(self.intent_head(x), dim=1)
        slots_sigms = torch.sigmoid(self.slots_head(x))
        return intent_probs,slots_sigms


class PolicyA2CAgent(nn.Module):
    def __init__(
        self, vocab_size, num_intents, num_slots, hidden_dim=64, embed_dim=32,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocab_size,hidden_dim,embed_dim)
        self.actor = Actor(hidden_dim,num_intents,num_slots)
        self.critic = nn.Linear(embed_dim, 1)

    def forward(self, x):
        features_pooled = self.encoder(x)
        intent_probs, slots_sigms = self.actor(features_pooled)
        return intent_probs, slots_sigms

    def step(self, state, draw=sample_from_distr):
        intent_probs, slot_sigms = self.forward(state)
        cd = Categorical(intent_probs)
        bd = Bernoulli(slot_sigms)
        intent, slots = draw(cd, bd)
        if len(intent.shape) == 1:  # cause its stupid!
            intent = intent.unsqueeze(0)
        log_prob = torch.sum(
            torch.cat([cd.log_prob(intent), bd.log_prob(slots)], dim=1)
        )
        return (intent.item(), slots.numpy()), log_prob


class PyTorchA2CPolicy(PyTorchReinforcePolicy):
    def __init__(
        self,
        ontology,
        database,
        agent_id=0,
        agent_role="system",
        domain=None,
        alpha=0.95,
        epsilon=0.95,
        gamma=0.99,
        alpha_decay=0.995,
        epsilon_decay=0.995,
        print_level="debug",
        epsilon_min=0.05,
        **kwargs
    ):
        super().__init__(
            ontology,
            database,
            agent_id,
            agent_role,
            domain,
            alpha,
            epsilon,
            gamma,
            alpha_decay,
            epsilon_decay,
            print_level,
            epsilon_min,
            **kwargs
        )

    @staticmethod
    def _calc_returns(exp, gamma):
        returns = []
        R = 0
        for log_prob, r in reversed(exp):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return returns

    def train(self, batch: List):
        self.agent.train()
        self.agent.to(DEVICE)
        losses = [
            loss.unsqueeze(0) for d in batch for loss in self._calc_dialogue_losses(d)
        ]
        policy_loss = torch.cat(losses).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.losses.append(float(policy_loss.data.cpu().numpy()))

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _calc_dialogue_losses(self, dialogue: List[Dict]):
        exp = []
        for turn in dialogue:
            x = self.encode_state(turn["state"]).to(DEVICE)
            action_encs = self.encode_action(turn["action"])
            action = tuple(
                [
                    torch.from_numpy(a).float().unsqueeze(0).to(DEVICE)
                    for a in action_encs
                ]
            )
            draw_method = lambda *_: action
            _, log_probs = self.agent.step(x, draw_method)
            exp.append((log_probs, turn["reward"]))
        returns = self._calc_returns(exp, self.gamma)
        dialogue_losses = [-log_prob * R for (log_prob, _), R in zip(exp, returns)]
        return dialogue_losses
