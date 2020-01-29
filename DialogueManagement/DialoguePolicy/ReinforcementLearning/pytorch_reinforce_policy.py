import os

import numpy
import random

from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

STATE_DIM = 45

class PolicyAgent(nn.Module):
    def __init__(self, obs_dim, num_actions,hidden_dim = 1024) -> None:
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

    def calc_probs(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def log_probs(self,state,action):
        probs = self.calc_probs(state)
        m = Categorical(probs)
        action_tensor = torch.from_numpy(action).float().unsqueeze(0)
        return m.log_prob(action_tensor)

class PyTorchReinforcePolicy(QPolicy):
    def __init__(self, ontology, database, agent_id=0, agent_role='system', domain=None,
                 alpha=0.95, epsilon=0.95, gamma=0.99, alpha_decay=0.995,
                 epsilon_decay=0.995, print_level='debug', epsilon_min=0.05):
        assert gamma == 0.99
        super().__init__(ontology, database, agent_id, agent_role, domain, alpha,
                         epsilon, gamma, alpha_decay, epsilon_decay, print_level,
                         epsilon_min)

        self.agent:PolicyAgent = PolicyAgent(STATE_DIM, self.NActions)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-2)

    def next_action(self, state:SlotFillingDialogueState):
        self.agent.eval()
        if (self.is_training and random.random() < self.epsilon):
            sys_acts = self.warmup_policy.next_action(state)
        else:
            state_enc = self.encode_state(state)
            action,_ = self.agent.step(numpy.array(state_enc,dtype=numpy.int64))
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

    def train(self, dialogues):
        self.agent.train()
        losses = []
        policy_losses = []
        for k,dialogue in enumerate(dialogues):
            exp = []
            for turn in dialogue:
                state_enc = self.encode_state(turn['state'])
                action_enc = self.encode_action(turn['action'])

                log_probs = self.agent.log_probs(numpy.array(state_enc),numpy.array(action_enc))
                exp.append((action_enc, log_probs, turn['reward']))

            returns = self._calc_returns(exp, self.gamma)
            policy_losses.extend([-log_prob * R for (_, log_prob, _), R in zip(exp, returns)])

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


