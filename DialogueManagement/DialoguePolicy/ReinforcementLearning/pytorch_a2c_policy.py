from typing import List, Dict, Tuple, NamedTuple

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
    state: SlotFillingDialogueState
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

    def train(self, batch: List):
        self.agent.train()
        self.agent.to(DEVICE)
        dialogues = [self._build_dialogue_turns(dialogue) for dialogue in batch]
        exps = [self._dialogue_to_experience(d) for d in dialogues]

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _build_dialogue_turns(self, dialogue: List[Dict]):
        x = [(d["action"], d["state"], d["reward"]) for d in dialogue]

        if any([not isinstance(d["action"], ValueDialogAct) for d in dialogue]):
            rewards = [t["reward"] for t in dialogue]
            returns = calc_discounted_returns(rewards, self.gamma)
            turns = [
                DialogTurn(ValueDialogAct(a.intent, a.params, v), s, r)
                for (a, s, r), v in zip(x, returns)
            ]
        else:
            turns = [DialogTurn(a, s, r) for a, s, r in x]
        return turns

    def _dialogue_to_experience(
        self, dialogue: List[DialogTurn]
    ) -> List[Tuple[EnvStep, AgentStep]]:
        exp = []

        for k, turn in enumerate(dialogue):
            observation = self.encode_state(turn.state).to(DEVICE)
            action_encs = self.encode_action([turn.act])
            action = tuple(
                [
                    torch.from_numpy(a).float().unsqueeze(0).to(DEVICE)
                    for a in action_encs
                ]
            )
            done = torch.from_numpy(np.array([k == len(dialogue) - 1]).astype(np.int))
            env_step = EnvStep(observation, turn.reward, done)
            agent_step = AgentStep(action, turn.act.value)
            exp.append((env_step, agent_step))
        return exp

    def decode_action(self, action_enc: Tuple):
        intent, slots = self.action_enc.decode(*action_enc)
        slots = self._filter_slots(intent, slots)
        return ValueDialogAct(
            intent, params=[DialogueActItem(slot, Operator.EQ, "") for slot in slots],
        )
