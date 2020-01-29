import os
import shutil
from os import chdir

import numpy
import random

from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

STATE_DIM = 57

class PolicyAgent(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=1024) -> None:
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def step(self, state):
        probs = self.calc_probs(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def calc_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def log_probs(self, state, action):
        probs = self.calc_probs(state)
        m = Categorical(probs)
        action_tensor = torch.from_numpy(action).float().unsqueeze(0)
        return m.log_prob(action_tensor)


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

        self.agent: PolicyAgent = PolicyAgent(STATE_DIM, self.NActions)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-2)

    def next_action(self, state: SlotFillingDialogueState):
        self.agent.eval()
        if self.is_training and random.random() < self.epsilon:
            if random.random() < 0.5:
                sys_acts = self.warmup_policy.next_action(state)
            else:
                sys_acts = self.decode_action(
                    random.choice(range(0, self.NActions)))

        else:
            state_enc = self.encode_state(state)
            action, _ = self.agent.step(numpy.array(state_enc, dtype=numpy.int64))
            sys_acts = self.decode_action(action)

        return sys_acts

    @staticmethod
    def _calc_returns(exp, gamma):
        returns = []
        R = 0
        for action, log_prob, r in reversed(exp):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return returns

    def encode_state(self, state):
        state_int = super().encode_state(state)
        state_enc = [int(b) for b in "{0:b}".format(state_int)]
        state_enc = [0 for _ in range(STATE_DIM - len(state_enc))] + state_enc
        return state_enc

    def train(self, dialogues):
        self.agent.train()
        losses = []
        policy_losses = []
        for k, dialogue in enumerate(dialogues):
            exp = []
            for turn in dialogue:
                state_enc = self.encode_state(turn['state'])
                assert len(state_enc)==STATE_DIM
                action_enc = self.encode_action(turn["action"])

                log_probs = self.agent.log_probs(
                    numpy.array(state_enc), numpy.array(action_enc)
                )
                exp.append((action_enc, log_probs, turn["reward"]))

            returns = self._calc_returns(exp, self.gamma)
            policy_losses.extend(
                [-log_prob * R for (_, log_prob, _), R in zip(exp, returns)]
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
    '''
    simple test
    '''
    def clean_dir(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)

    num_dialogues = 32
    base_path = '../../../../alex-plato/experiments/exp_04'

    chdir('%s' % base_path)

    clean_dir('logs')
    clean_dir('policies')
    if os.path.isfile('/tmp/agent'):
        os.remove('/tmp/agent')

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
