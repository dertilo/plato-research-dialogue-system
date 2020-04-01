import os
import re
import shutil
from os import chdir
from typing import List

import numpy
import random

from sklearn import preprocessing
from torchtext.data import Field, Example

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from DialogueManagement.DialoguePolicy.dialogue_common import create_random_dialog_act, \
    Domain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyAgent(nn.Module):
    def __init__(self, vocab_size, num_actions, hidden_dim=64, embed_dim=32,) -> None:
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
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.affine2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        features = self.convnet(x)
        features_pooled = self.pooling(features).squeeze(2)
        return F.softmax(self.affine2(features_pooled), dim=1)

    def step(self, state):
        probs = self.calc_probs(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def calc_probs(self, state):
        return self.forward(state)

    def log_probs(self, state, action):
        probs = self.calc_probs(state)
        m = Categorical(probs)
        action_tensor = torch.from_numpy(action).float().unsqueeze(0)
        return m.log_prob(action_tensor)


import json


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
        gamma = 0.99
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
        vocab_size = len(self.text_field.vocab)

        self.action_enc=self._build_action_encoder(self.domain)
        self.NActions = self.action_enc.classes_.shape[0]

        self.agent: PolicyAgent = PolicyAgent(vocab_size, self.NActions)
        print(self.agent)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-2)

    @staticmethod
    def _build_text_field(domain:Domain):
        tokens = [
            v
            for vv in domain._asdict().values()
            if isinstance(vv, list)
            for v in vv
        ]
        special_tokens = [str(k) for k in range(10)] + [
            ".",
            ",",
            ";",
            ":",
            '"',
            "'",
            "{",
            "}",
            "[",
            "]",
            "(",
            ")",
        ]

        def regex_tokenizer(text, pattern=r"(?u)(?:\b\w\w+\b|\S)") -> List[str]:
            return [m.group() for m in re.finditer(pattern, text)]

        text_field = Field(batch_first=True, tokenize=regex_tokenizer)
        text_field.build_vocab(tokens + special_tokens)
        return text_field

    @staticmethod
    def _build_action_encoder(domain:Domain):
        action_enc = preprocessing.LabelEncoder()
        informs = [json.dumps({"inform": [x]}) for x in domain.requestable_slots]
        requests = [
            json.dumps({"request": [x]}) for x in domain.system_requestable_slots
        ]
        action_enc.fit(
            [
                [x]
                for x in informs
                         + requests
                         + [json.dumps({s: []}) for s in domain.dstc2_acts_sys]
            ]
        )
        return action_enc

    def next_action(self, state: SlotFillingDialogueState):
        self.agent.eval()
        if self.is_training and random.random() < self.epsilon:
            if random.random() < 1.0:
                sys_acts = self.warmup_policy.next_action(state)
            else:
                sys_acts = create_random_dialog_act(self.domain, is_system=True)

        else:
            state_enc = self.encode_state(state)
            action, _ = self.agent.step(state_enc)
            sys_acts = self.decode_action(action)

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
        return self.text_field.numericalize([example.dialog_state])

    def _get_dialog_act_slots(self, act: DialogueAct):
        if act.params is not None and act.intent in self.domain.acts_params:
            slots = [d.slot for d in act.params]
        else:
            slots = []
        return slots

    def encode_action(self, acts: List[DialogueAct], system=True) -> str:
        # TODO(tilo): DialogueManager makes offer with many informs, these should not be encoded here!
        if any([a.intent == "offer" for a in acts]):
            acts = acts[:1]
        assert len(acts) == 1
        d = {act.intent: self._get_dialog_act_slots(act) for act in acts}
        jsoned = json.dumps(d)
        return self.action_enc.transform([[jsoned]])

    def decode_action(self, action_enc):
        x = self.action_enc.inverse_transform([action_enc])
        dicts = {k: v for d in (json.loads(s) for s in x) for k, v in d.items()}
        acts = [
            DialogueAct(
                intent,
                params=[DialogueActItem(slot, Operator.EQ, "") for slot in slots],
            )
            for intent, slots in dicts.items()
        ]
        return acts

    def train(self, dialogues):
        self.agent.train()
        losses = []
        policy_losses = []
        for k, dialogue in enumerate(dialogues):
            exp = []
            for turn in dialogue:
                x = self.encode_state(turn["state"])
                # assert len(state_str)==STATE_DIM
                action_enc = self.encode_action(turn["action"])
                log_probs = self.agent.log_probs(x, numpy.array(action_enc))
                exp.append((log_probs, turn["reward"]))

            returns = self._calc_returns(exp, self.gamma)
            policy_losses.extend(
                [-log_prob * R for (log_prob, _), R in zip(exp, returns)]
            )

        policy_loss = torch.cat(policy_losses).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        losses.append(policy_loss.data.numpy())

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path=None):
        torch.save(self.agent.state_dict(), path)
        # self.agent=None
        # pickle.dump(self,path+'/pytorch_policy.pkl')

    def load(self, path=None):
        if os.path.isfile(path):
            agent = PolicyAgent(STATE_DIM, self.NActions)
            agent.load_state_dict(torch.load(path))
            self.agent = agent


if __name__ == "__main__":
    """
    simple test
    """

    def clean_dir(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)

    num_dialogues = 32
    base_path = "../../../../alex-plato/experiments/exp_09"

    chdir("%s" % base_path)

    clean_dir("logs")
    clean_dir("policies")
    if os.path.isfile("/tmp/agent"):
        os.remove("/tmp/agent")

    config = {
        "GENERAL": {
            "print_level": "info",
            "interaction_mode": "simulation",
            "agents": 1,
            "runs": 5,
            "experience_logs": {
                "save": False,
                "load": False,
                "path": "logs/train_reinforce_logs.pkl",
            },
        },
        "DIALOGUE": {
            "num_dialogues": 1000,
            "initiative": "system",
            "domain": "CamRest",
            "ontology_path": "domain/alex-rules.json",
            "db_path": "domain/alex-dbase.db",
            "db_type": "sql",
            "cache_sql_results": True,
        },
        "AGENT_0": {
            "role": "system",
            "USER_SIMULATOR": {
                "simulator": "agenda",
                "patience": 5,
                "pop_distribution": [1.0],
                "slot_confuse_prob": 0.0,
                "op_confuse_prob": 0.0,
                "value_confuse_prob": 0.0,
            },
            "DM": {
                "policy": {
                    "type": "pytorch_reinforce",
                    "train": True,
                    "learning_rate": 0.01,
                    "learning_decay_rate": 0.995,
                    "discount_factor": 0.8,
                    "exploration_rate": 1.0,
                    "exploration_decay_rate": 0.95,
                    "min_exploration_rate": 0.01,
                    "policy_path": "/tmp/agent",
                }
            },
            "NLU": None,
            "DST": {"dst": "dummy"},
            "NLG": None,
        },
    }
    from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent

    ca = ConversationalSingleAgent(config)
    ca.initialize()
    ca.minibatch_length = 8
    ca.train_epochs = 1
    ca.train_interval = 8

    for dialogue in range(num_dialogues):

        if (dialogue + 1) % 10 == 0:
            print("=====================================================")
            print("Dialogue %d (out of %d)\n" % (dialogue + 1, num_dialogues))

        ca.start_dialogue()

        while not ca.terminated():
            ca.continue_dialogue()

        ca.end_dialogue()

    # Collect statistics
    statistics = {"AGENT_0": {}}
    statistics["AGENT_0"]["dialogue_success_percentage"] = 100 * float(
        ca.num_successful_dialogues / num_dialogues
    )
    statistics["AGENT_0"]["avg_cumulative_rewards"] = float(
        ca.cumulative_rewards / num_dialogues
    )
    statistics["AGENT_0"]["avg_turns"] = float(ca.total_dialogue_turns / num_dialogues)
    statistics["AGENT_0"]["objective_task_completion_percentage"] = 100 * float(
        ca.num_task_success / num_dialogues
    )

    print(
        "\n\nDialogue Success Rate: {0}\nAverage Cumulative Reward: {1}"
        "\nAverage Turns: {2}".format(
            statistics["AGENT_0"]["dialogue_success_percentage"],
            statistics["AGENT_0"]["avg_cumulative_rewards"],
            statistics["AGENT_0"]["avg_turns"],
        )
    )
