import abc
from typing import Dict, Any, NamedTuple, Tuple, List, Union
import torch
import torch.nn as nn

from DialogueManagement.DialoguePolicy.ReinforcementLearning.rlutil.dictlist import (
    DictList,
)
from DialogueManagement.DialoguePolicy.ReinforcementLearning.rlutil.experience_memory import (
    ExperienceMemory,
    fill_with_zeros,
)


def flatten_parallel_rollout(d):
    return {
        k: flatten_parallel_rollout(v) if isinstance(v, dict) else flatten_array(v)
        for k, v in d.items()
    }


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])


class EnvStep(NamedTuple):
    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class AgentStep(NamedTuple):
    actions: Union[Dict,Tuple]
    v_values: torch.Tensor


class EnvStepper:
    @abc.abstractmethod
    def step(self, agent_step: AgentStep) -> EnvStep:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> EnvStep:
        raise NotImplementedError


class AgentStepper:
    @abc.abstractmethod
    def step(self, observation: torch.Tensor) -> AgentStep:
        raise NotImplementedError


class Rollout(NamedTuple):
    env_steps: EnvStep
    agent_steps: AgentStep
    advantages: torch.FloatTensor
    returnn: torch.FloatTensor


class AbstractA2CAgent(nn.Module, AgentStepper):

    @abc.abstractmethod
    def calc_distr_value(self,obs):
        raise NotImplementedError


def generalized_advantage_estimation(
    rewards, values, dones, num_rollout_steps, discount, gae_lambda
):
    assert values.shape[0] == 1 + num_rollout_steps
    advantage_buffer = torch.zeros(rewards.shape[0] - 1, rewards.shape[1])
    next_advantage = 0
    for i in reversed(range(num_rollout_steps)):
        mask = torch.tensor((1 - dones[i + 1]), dtype=torch.float32)
        bellman_delta = rewards[i + 1] + discount * values[i + 1] * mask - values[i]
        advantage_buffer[i] = (
            bellman_delta + discount * gae_lambda * next_advantage * mask
        )
        next_advantage = advantage_buffer[i]
    return advantage_buffer


class A2CParams(NamedTuple):
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_rollout_steps: int = 4
    discount: float = 0.99
    lr: float = 1e-2
    gae_lambda: float = 0.95


def calc_loss(exps: Rollout, agent: AbstractA2CAgent, p: A2CParams):
    dist, value = agent.calc_distr_value(exps.env_steps.observation)
    entropy = dist.entropy().mean()
    policy_loss = -(dist.log_prob(**exps.agent_steps.actions) * exps.advantages).mean()
    value_loss = (value - exps.returnn).pow(2).mean()
    loss = policy_loss - p.entropy_coef * entropy + p.value_loss_coef * value_loss
    return loss


def collect_experiences_calc_advantage(
    exp_mem: ExperienceMemory, params: A2CParams
) -> Rollout:
    assert exp_mem.last_written_idx == params.num_rollout_steps

    env_steps = exp_mem.buffer.env
    agent_steps = exp_mem.buffer.agent
    advantages = generalized_advantage_estimation(
        rewards=env_steps.reward,
        values=agent_steps.v_values,
        dones=env_steps.done,
        num_rollout_steps=params.num_rollout_steps,
        discount=params.discount,
        gae_lambda=params.gae_lambda,
    )
    return Rollout(
        **{
            "env_steps": DictList(**flatten_parallel_rollout(env_steps[:-1])),
            "agent_steps": DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            "advantages": flatten_array(advantages),
            "returnn": flatten_array(agent_steps[:-1].v_values + advantages),
        }
    )


def build_experience_memory(steps: List[Dict], rollout_len=5) -> ExperienceMemory:
    num_steps = rollout_len +1 #+1 cause the very first is the "initial-step"
    windows = [
        steps[i : (i + num_steps)]
        for i in range(0, (len(steps) // num_steps) * num_steps, num_steps)
    ]
    expmem = None
    for k in range(num_steps):
        dictlist = fill_with_zeros(len(windows), steps[0])
        for i, exp in enumerate(windows):
            dictlist[i] = exp[k]
        if expmem is None:
            expmem = ExperienceMemory(num_steps, dictlist)
        else:
            expmem.store_single(dictlist)
    return expmem
