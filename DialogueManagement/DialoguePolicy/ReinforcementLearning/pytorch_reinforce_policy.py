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

from DialogueManagement.DialoguePolicy.dialogue_common import (
    create_random_dialog_act,
    Domain,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_from_distr(*distrs):
    return tuple([d.sample() for d in distrs])


class PolicyAgent(nn.Module):
    def __init__(
        self, vocab_size, num_intents, num_slots, hidden_dim=64, embed_dim=32,
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

        self.intent_head = nn.Linear(hidden_dim, num_intents)

        self.slots_head = nn.Linear(hidden_dim, num_slots)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        features = self.convnet(x)
        features_pooled = self.pooling(features).squeeze(2)
        intent_probs = F.softmax(self.intent_head(features_pooled), dim=1)
        slots_sigms = torch.sigmoid(self.slots_head(features_pooled))
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


class ActionEncoder:
    def __init__(self, domain: Domain) -> None:
        self.intent_enc = preprocessing.LabelEncoder()
        intents = set(domain.acts_params + domain.dstc2_acts_sys)
        self.intent_enc.fit([[x] for x in intents])

        self.slot_enc = preprocessing.MultiLabelBinarizer()
        slots = set(domain.requestable_slots + domain.system_requestable_slots)
        self.slot_enc.fit([[x] for x in slots])

    def encode(self, intent: str, slots: List[str]):
        intent_enc = self.intent_enc.transform([intent])
        slots_enc = self.slot_enc.transform([slots])[0]
        return intent_enc, slots_enc

    def decode(self, intent_enc, slots_enc):
        return (
            self.intent_enc.inverse_transform([intent_enc])[0],
            self.slot_enc.inverse_transform(slots_enc)[0],
        )


class PyTorchReinforcePolicy(QPolicy):
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
        assert gamma>0.9
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
        )

        self.text_field = self._build_text_field(self.domain)
        self.vocab_size = len(self.text_field.vocab)

        self.action_enc = ActionEncoder(self.domain)
        self.NActions = None

        self.PolicyAgentModelClass = kwargs.get("PolicyAgentModelClass", PolicyAgent)
        self.num_intents = len(self.domain.acts_params) + len(
            self.domain.dstc2_acts_sys
        )
        self.num_slots = len(
            set(self.domain.system_requestable_slots + self.domain.requestable_slots)
        )
        self.agent = self.PolicyAgentModelClass(
            self.vocab_size, self.num_intents, self.num_slots
        )
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-2)
        self.losses = []

    def _build_text_field(self, domain: Domain):
        dings = super().encode_state(SlotFillingDialogueState([]))
        tokens = [
            v for vv in domain._asdict().values() if isinstance(vv, list) for v in vv
        ]

        def regex_tokenizer(text, pattern=r"(?u)(?:\b\w\w+\b|\S)") -> List[str]:
            return [m.group() for m in re.finditer(pattern, text)]

        state_tokens = [t for t in regex_tokenizer(dings) if t != '"']
        special_tokens = [str(k) for k in range(10)] + state_tokens
        text_field = Field(batch_first=True, tokenize=regex_tokenizer)
        text_field.build_vocab([tokens + special_tokens])
        return text_field

    def next_action(self, state: SlotFillingDialogueState):
        self.agent.eval()
        self.agent.to(DEVICE)
        if self.is_training and random.random() < self.epsilon:
            if random.random() < 1.0:
                sys_acts = self.warmup_policy.next_action(state)
            else:
                sys_acts = create_random_dialog_act(self.domain, is_system=True)

        else:
            state_enc = self.encode_state(state)
            action, _ = self.agent.step(state_enc.to(DEVICE))
            sys_acts = [self.decode_action(action)]

        return sys_acts

    @staticmethod
    def _calc_returns(exp, gamma):
        returns = []
        R = 0
        for log_prob, r in reversed(exp):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return returns

    def encode_state(self, state: SlotFillingDialogueState) -> torch.LongTensor:
        state_string = super().encode_state(state)
        example = Example.fromlist([state_string], [("dialog_state", self.text_field)])
        tokens = [t for t in example.dialog_state if t in self.text_field.vocab.stoi]
        return self.text_field.numericalize([tokens])

    def _get_dialog_act_slots(self, act: DialogueAct):
        if act.params is not None and act.intent in self.domain.acts_params:
            slots = [d.slot for d in act.params]
        else:
            slots = []
        return slots

    def encode_action(
        self, acts: List[DialogueAct], system=True
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # TODO(tilo): DialogueManager makes offer with many informs, these should not be encoded here!
        if any([a.intent == "offer" for a in acts]):
            acts = acts[:1]
        assert len(acts) == 1
        slots = [p.slot for p in acts[0].params]
        encoded_action = self.action_enc.encode(acts[0].intent, slots)
        return encoded_action

    def decode_action(self, action_enc: Tuple):
        intent, slots = self.action_enc.decode(*action_enc)
        slots = self._filter_slots(intent, slots)
        return DialogueAct(
            intent, params=[DialogueActItem(slot, Operator.EQ, "") for slot in slots],
        )

    def _filter_slots(self, intent, slots):
        if intent == "inform":
            slots = filter(lambda s: s in self.domain.requestable_slots, slots)
        elif intent == "request":
            slots = filter(lambda s: s in self.domain.system_requestable_slots, slots)
        else:
            slots = []
        return slots

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

    def save(self, path=None):
        torch.save(self.agent.state_dict(), path)
        # self.agent=None
        # pickle.dump(self,path+'/pytorch_policy.pkl')

    def load(self, path=None):
        if os.path.isfile(path):
            agent = self.PolicyAgentModelClass(
                self.vocab_size, self.num_intents, self.num_slots
            )
            agent.load_state_dict(torch.load(path))
            self.agent = agent
