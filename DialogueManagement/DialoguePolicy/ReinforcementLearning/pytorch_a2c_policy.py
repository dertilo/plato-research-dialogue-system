import random
from typing import List, Dict, NamedTuple

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
import torch
import torch.nn as nn

from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_common import (
    StateEncoder,
    CommonDistribution,
    calc_discounted_returns,
    tokenize,
    Actor,
)
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_reinforce_policy import (
    PyTorchReinforcePolicy,
)
from DialogueManagement.DialoguePolicy.ReinforcementLearning.rlutil.advantage_actor_critic import (
    EnvStep,
    AgentStep,
    AbstractA2CAgent,
    A2CParams,
    calc_loss,
    build_experience_memory,
    collect_experiences_calc_advantage,
)
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyA2CAgent(AbstractA2CAgent):
    def __init__(
        self,
        vocab_size,
        num_intents,
        num_slots,
        encode_dim=64,
        embed_dim=32,
        padding_idx=None,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocab_size, encode_dim, embed_dim, padding_idx)
        self.actor = Actor(encode_dim, num_intents, num_slots)
        self.critic = nn.Linear(encode_dim, 1)

    def forward(self, x):
        features_pooled = self.encoder(x)
        intent_probs, slots_sigms = self.actor(features_pooled)
        value = self.critic(features_pooled)
        return (intent_probs, slots_sigms), value

    def calc_value(self, x):
        value = self.critic(self.encoder(x))
        return value

    def calc_distr_value(self, state):
        (intent_probs, slot_sigms), value = self.forward(state)
        distr = CommonDistribution(intent_probs, slot_sigms)
        return distr, value

    def calc_distr(self, state):
        distr, value = self.calc_distr_value(state)
        return distr

    def step(self, x) -> AgentStep:
        (intent_probs, slot_sigms), value = self.forward(x)
        distr = CommonDistribution(intent_probs, slot_sigms)
        intent, slots = distr.sample()
        v_values = value.data
        return AgentStep((intent.item(), slots.numpy()), v_values)


class DialogTurn(NamedTuple):
    act: DialogueAct
    tokens: List[str]
    reward: float
    state_value: float


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

        self.a2c_params = A2CParams()

    def get_policy_agent_model_class(self, kwargs):
        return kwargs.get("PolicyAgentModelClass", PolicyA2CAgent)

    def next_action(self, state: SlotFillingDialogueState):
        self.agent.eval()
        self.agent.to(DEVICE)

        state_enc = self.encode_state(state).to(DEVICE)
        with torch.no_grad():
            if self.is_training and random.random() < self.epsilon:
                warmup_acts = self.warmup_policy.next_action(state)
                sys_acts = warmup_acts
                value = self.agent.calc_value(state_enc).cpu().item()
            else:
                agent_step = self.agent.step(state_enc)
                value = agent_step.v_values.cpu().item()
                sys_acts = [self.decode_action(agent_step)]
            state.value = value

        return sys_acts

    def _calc_loss(self, batch: List[List[Dict]]):
        batch_of_turns = [self.process_dialogue_to_turns(dialogue=d) for d in batch]

        sequences = [t.tokens for turns in batch_of_turns for t in turns]
        max_seq_len = max([len(s) for s in sequences])
        self.text_field.fix_length = max_seq_len

        steps = [e for d in batch_of_turns for e in self._dialogue_to_steps(d)]
        expmem = build_experience_memory(steps, self.a2c_params.num_rollout_steps)
        rollout = collect_experiences_calc_advantage(expmem, self.a2c_params)

        loss = calc_loss(rollout, self.agent, self.a2c_params)
        return loss

    def process_dialogue_to_turns(self, dialogue: List[Dict]) -> List[DialogTurn]:
        assert dialogue[0]["action"][0].intent == "welcomemsg"
        assert dialogue[-1]["action"][0].intent == "bye"
        dialogue[-2]["reward"] = dialogue[-1]["reward"]
        dialogue = dialogue[1:-1]
        rewards = [t["reward"] for t in dialogue]
        returns = calc_discounted_returns(rewards, self.gamma)
        turns = [
            DialogTurn(
                d["action"][0],
                tokenize(self.text_field, d["state"]),
                d["reward"],
                d["state"].value,
            )
            for d, ret in zip(dialogue, returns)
            if d["state"].value is not None
        ]
        return turns

    def _dialogue_to_steps(self, dialogue: List[DialogTurn]) -> List[Dict]:
        steps = []

        for k, turn in enumerate(dialogue):
            observation = self.text_field.process([turn.tokens]).squeeze().to(DEVICE)
            action_encs = self.encode_action([turn.act])
            action = {
                n: torch.from_numpy(a).float().to(DEVICE)
                for n, a in zip(["intent", "slots"], action_encs)
            }
            done = torch.from_numpy(
                np.array([k == len(dialogue) - 1]).astype(np.int)
            ).squeeze()
            env_step = EnvStep(
                observation,
                torch.from_numpy(np.array(turn.reward)).type(torch.float32),
                done,
            )
            agent_step = AgentStep(
                action, torch.from_numpy(np.array(turn.state_value)).type(torch.float32)
            )
            steps.append({"env": env_step._asdict(), "agent": agent_step._asdict()})
        return steps

    def decode_action(self, step: AgentStep):
        intent, slots = self.action_enc.decode(*step.actions)
        slots = self._filter_slots(intent, slots)
        return DialogueAct(
            intent, params=[DialogueActItem(slot, Operator.EQ, "") for slot in slots],
        )
