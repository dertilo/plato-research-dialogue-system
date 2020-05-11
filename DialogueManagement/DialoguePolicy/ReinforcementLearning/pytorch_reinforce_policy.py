import os
import re
from collections import Counter
from typing import List, Dict, Tuple, NamedTuple

import numpy
import random

from sklearn import preprocessing
from torchtext.data import Field, Example

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
import torch
import torch.nn as nn
import torch.optim as optim
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_common import (
    StateEncoder,
    CommonDistribution,
    process_dialogue_to_turns,
    Actor,
    DEVICE)
from DialogueManagement.DialoguePolicy.dialogue_common import (
    Domain,
    state_to_json,
)


class AgentStep(NamedTuple):
    action: Tuple[int, numpy.ndarray]


class PolicyAgent(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_intents,
        num_slots,
        hidden_dim=64,
        embed_dim=32,
        padding_idx=None,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocab_size, hidden_dim, embed_dim, padding_idx)
        self.actor = Actor(hidden_dim, num_intents, num_slots)

    def forward(self, x):
        features_pooled = self.encoder(x)
        intent_probs, slots_sigms = self.actor(features_pooled)
        return intent_probs, slots_sigms

    def calc_distr(self, state):
        intent_probs, slot_sigms = self.forward(state)
        distr = CommonDistribution(intent_probs, slot_sigms)
        return distr

    def step(self, state) -> AgentStep:
        distr = self.calc_distr(state)
        intent, slots = distr.sample()
        return AgentStep((intent.cpu().item(), slots.cpu().numpy()))


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
        decoded_intent = self.intent_enc.inverse_transform([intent_enc])[0]
        decoded_slots = self.slot_enc.inverse_transform(slots_enc)[0]

        return (
            decoded_intent,
            decoded_slots,
        )


class PyTorchReinforcePolicy(QPolicy):
    def __init__(
        self,
        ontology,
        database,
        agent_id=0,
        agent_role="system",
        domain=None,
        epsilon=0.95,
        gamma=0.99,
        epsilon_decay=0.995,
        print_level="debug",
        epsilon_min=0.05,
        **kwargs
    ):
        assert gamma > 0.9
        super().__init__(
            ontology,
            database,
            agent_id,
            agent_role,
            domain,
            None,
            epsilon,
            gamma,
            None,
            epsilon_decay,
            print_level,
            epsilon_min,
        )

        self.text_field = self._build_text_field(self.domain)
        self.vocab_size = len(self.text_field.vocab)
        self.num_pos = 0
        self.action_enc = ActionEncoder(self.domain)
        self.NActions = None

        self.num_intents = len(self.domain.acts_params) + len(
            self.domain.dstc2_acts_sys
        )
        self.num_slots = len(
            set(self.domain.system_requestable_slots + self.domain.requestable_slots)
        )
        self.PolicyAgentModelClass = self.get_policy_agent_model_class(kwargs)
        self.agent = self.PolicyAgentModelClass(
            self.vocab_size,
            self.num_intents,
            self.num_slots,
            padding_idx=self.text_field.vocab.stoi["<pad>"],
        )
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-3)
        self.losses = []
        self.counter = {"warmup":Counter(),"learned":Counter()}

    def get_policy_agent_model_class(self, kwargs):
        return kwargs.get("PolicyAgentModelClass", PolicyAgent)

    def _build_text_field(self, domain: Domain):
        dings = state_to_json(SlotFillingDialogueState([]))
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
            sys_acts = self.warmup_policy.next_action(state)
            self.counter["warmup"].update({len(sys_acts[0].params):1})
        else:
            state_enc = self.encode_state(state)
            with torch.no_grad():
                agent_step = self.agent.step(state_enc.to(DEVICE))
            sys_acts = [self.decode_action(agent_step)]
            self.counter["learned"].update({len(sys_acts[0].params):1})

        if not sys_acts or len(sys_acts) == 0:
            raise Exception('At least least one action has to be selected!')

        return sys_acts

    def encode_state(self, state: SlotFillingDialogueState) -> torch.LongTensor:
        state_string = state_to_json(state)
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
        return self.action_enc.encode(acts[0].intent, slots)

    def decode_action(self, step: AgentStep) -> DialogueAct:
        intent, slots = self.action_enc.decode(*step.action)
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

    def train(self, batch: List[List[Dict]]):
        self.num_pos = sum([1 for d in batch if d[-1]['success']])

        batch = self._balance_batch(batch)# TODO(tilo): not yet sure whether it is really necessary
        if len(batch)==0:
            return # no training will happen
        self.agent.train()
        self.agent.to(DEVICE)

        loss = self._calc_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 100.0)
        self.optimizer.step()
        self.losses.append(float(loss.data.cpu().numpy()))

        # Decay exploration rate
        if self.epsilon > self.epsilon_min and not self.warm_up_mode:
            self.epsilon *= self.epsilon_decay

    def _balance_batch(self, batch):
        '''
        at least half of the batch must have a positive reward
        examples with negative rewards are discarded
        '''
        self.num_pos = sum([1 for d in batch if d[-1]['reward']>0])
        batch_pos = [d for d in batch if d[-1]['reward']>0]
        batch_neg = [d for d in batch if d[-1]['reward']<0]
        random.shuffle(batch_neg)
        batch = batch_pos# + batch_neg[:(round(self.num_pos) - 1)]
        return batch

    def _calc_loss(self, batch: List[List[Dict]]):
        action, returns, x = self.prepare_batch(batch)
        distr = self.agent.calc_distr(x)
        losses = -distr.log_prob(*action) * returns
        return losses.mean()

    def prepare_batch(self, batch):
        turns = [
            t
            for dialogue in batch
            for t in process_dialogue_to_turns(self.text_field, dialogue, self.gamma)
        ]
        sequences = [turn.tokens for turn in turns]
        max_seq_len = max([len(s) for s in sequences])
        self.text_field.fix_length = max_seq_len
        state_enc = self.text_field.process(sequences).squeeze().to(DEVICE)
        action_encs = [self.encode_action([turn.act]) for turn in turns]
        action = tuple(
            torch.from_numpy(numpy.array(a)).float().to(DEVICE)
            for a in zip(*action_encs)
        )

        returns = torch.from_numpy(
            numpy.array([t.returnn for t in turns], dtype=numpy.float32)
        ).to(DEVICE)

        return action, returns, state_enc

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
