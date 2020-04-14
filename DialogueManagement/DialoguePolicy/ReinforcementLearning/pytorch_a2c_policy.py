from typing import List, Dict, Tuple, NamedTuple

from torchtext.data import Example

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_common import (
    StateEncoder,
    CommonDistribution,
    calc_discounted_returns,
)
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_reinforce_policy import (
    PyTorchReinforcePolicy,
)
from DialogueManagement.DialoguePolicy.ReinforcementLearning.rlutil.advantage_actor_critic import (
    EnvStep,
    AgentStep,
    AgentStepper,
    AbstractA2CAgent,
)
import numpy as np

from DialogueManagement.DialoguePolicy.ReinforcementLearning.rlutil.experience_memory import (
    fill_with_zeros,
    ExperienceMemory,
)
from DialogueManagement.DialoguePolicy.dialogue_common import state_to_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueDialogAct(DialogueAct):
    def __init__(self, intent="", params=None, value: float = 0.0):
        super().__init__(intent, params)
        self.value = value


class Actor(nn.Module):
    def __init__(self, hidden_dim, num_intents, num_slots) -> None:
        super().__init__()
        self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.slots_head = nn.Linear(hidden_dim, num_slots)

    def forward(self, x):
        intent_probs = F.softmax(self.intent_head(x), dim=1)
        slots_sigms = torch.sigmoid(self.slots_head(x))
        return intent_probs, slots_sigms


class PolicyA2CAgent(AbstractA2CAgent):
    def __init__(
        self, vocab_size, num_intents, num_slots, hidden_dim=64, embed_dim=32,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocab_size, hidden_dim, embed_dim)
        self.actor = Actor(hidden_dim, num_intents, num_slots)
        self.critic = nn.Linear(embed_dim, 1)

    def forward(self, x):
        features_pooled = self.encoder(x)
        intent_probs, slots_sigms = self.actor(features_pooled)
        return intent_probs, slots_sigms

    def calc_distr_value(self, state):
        intent_probs, slot_sigms = self.forward(state)
        distr = CommonDistribution(intent_probs, slot_sigms)
        value = self.critic(state)
        return distr, value

    def step(self, env_step: EnvStep, draw=lambda distr: distr.sample()) -> AgentStep:
        x = env_step.observation
        intent_probs, slot_sigms = self.forward(x)
        distr = CommonDistribution(intent_probs, slot_sigms)
        intent, slots = draw(distr)
        v_values = self.critic(x).data
        return AgentStep((intent.item(), slots.numpy()), v_values)


class DialogTurn(NamedTuple):
    act: ValueDialogAct
    tokenized_state_json: List[str]
    reward: float


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

        self.PolicyAgentModelClass = kwargs.get("PolicyAgentModelClass", PolicyA2CAgent)
        self.agent: AbstractA2CAgent = self.PolicyAgentModelClass(
            self.vocab_size, self.num_intents, self.num_slots
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

    def tokenize(self, state: SlotFillingDialogueState):
        state_string = state_to_json(state)
        example = Example.fromlist([state_string], [("dialog_state", self.text_field)])
        tokens = [t for t in example.dialog_state if t in self.text_field.vocab.stoi]
        return tokens

    def train(self, batch: List):
        self.agent.train()
        self.agent.to(DEVICE)
        dialogues = [self._build_dialogue_turns(dialogue) for dialogue in batch]
        seq_lenghts = [len(turn.tokenized_state_json) for d in dialogues for turn in d]
        max_seq_len_idx = np.argmax(seq_lenghts)
        max_seq_len = seq_lenghts[max_seq_len_idx]
        self.text_field.fix_length = max_seq_len

        exps = [e for d in dialogues for e in self._dialogue_to_experience(d)]
        w = 5
        windows = [exps[i : (i + w)] for i in range(0, (len(exps) // w) * w, w)]

        expmem = None
        for k in range(w):
            d = fill_with_zeros(len(windows), exps[0])
            for i, exp in enumerate(windows):
                d[i] = exp[k]
            if expmem is None:
                expmem = ExperienceMemory(w, d)
            else:
                expmem.store_single(d)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _build_dialogue_turns(self, dialogue: List[Dict]):
        x = [(d["action"], d["state"], d["reward"]) for d in dialogue]

        if any([not isinstance(d["action"], ValueDialogAct) for d in dialogue]):
            rewards = [t["reward"] for t in dialogue]
            returns = calc_discounted_returns(rewards, self.gamma)
            turns = [
                DialogTurn(
                    ValueDialogAct(a[0].intent, a[0].params, v), self.tokenize(s), r
                )
                for (a, s, r), v in zip(x, returns)
            ]
        else:
            turns = [DialogTurn(a[0], self.tokenize(s), r) for a, s, r in x]
        return turns

    def _dialogue_to_experience(self, dialogue: List[DialogTurn]) -> List[Dict]:
        exp = []

        for k, turn in enumerate(dialogue):
            observation = self.text_field.process([turn.tokenized_state_json]).squeeze().to(DEVICE)
            action_encs = self.encode_action([turn.act])
            action = {
                n: torch.from_numpy(a).float().to(DEVICE)
                for n, a in zip(["intent", "slots"], action_encs)
            }
            done = torch.from_numpy(
                np.array([k == len(dialogue) - 1]).astype(np.int)
            ).squeeze()
            env_step = EnvStep(
                observation, torch.from_numpy(np.array(turn.reward)), done
            )
            agent_step = AgentStep(action, torch.from_numpy(np.array(turn.act.value)))
            exp.append({"env": env_step._asdict(), "agent": agent_step._asdict()})
        return exp

    def decode_action(self, action_enc: Tuple):
        intent, slots = self.action_enc.decode(*action_enc)
        slots = self._filter_slots(intent, slots)
        return ValueDialogAct(
            intent, params=[DialogueActItem(slot, Operator.EQ, "") for slot in slots],
        )
