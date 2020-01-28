import numpy

from Dialogue.State import SlotFillingDialogueState
from DialogueManagement.DialoguePolicy.dialogue_common import encode_state, STATE_DIM, \
    encode_action, decode_action, Domain
from .. import DialoguePolicy, HandcraftedPolicy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from copy import deepcopy
from itertools import compress

import pickle
import random
import pprint
import os.path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# create logger
module_logger = logging.getLogger(__name__)


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


class PyTorchReinforcePolicy(DialoguePolicy.DialoguePolicy):
    def __init__(self, ontology, database, agent_id=0, agent_role='system',
                 domain=None, alpha=0.95, epsilon=0.95,
                 gamma=0.99, alpha_decay=0.995, epsilon_decay=0.995, print_level='info', epsilon_min=0.05):
        """
        Initialize parameters and internal structures

        :param ontology: the domain's ontology
        :param database: the domain's database
        :param agent_id: the agent's id
        :param agent_role: the agent's role
        :param alpha: the learning rate
        :param gamma: the discount rate
        :param epsilon: the exploration rate
        :param alpha_decay: the learning rate discount rate
        :param epsilon_decay: the exploration rate discount rate
        """
        assert agent_role == 'system'
        assert domain == 'CamRest'
        self.logger = logging.getLogger(__name__)
        self.warmup_mode = True
        self.print_level = print_level

        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.is_training = False
        self.IS_GREEDY_POLICY = True

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.database = database


        self.pp = pprint.PrettyPrinter(width=160)  # For debug!

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        self.warmup_policy = \
            HandcraftedPolicy.HandcraftedPolicy(self.ontology)


        # Extract lists of slots that are frequently used
        self.informable_slots = \
            deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = \
            deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        self.dstc2_acts = None

        self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm',
                               'deny', 'ack', 'bye', 'reqmore',
                               'welcomemsg', 'expl-conf', 'select',
                               'repeat', 'confirm-domain',
                               'confirm']

        # Does not include inform and request that are modelled
        # together with their arguments
        self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack',
                               'thankyou', 'bye', 'reqmore',
                               'hello', 'expl-conf', 'repeat',
                               'reqalts', 'restart', 'confirm']

        self.dstc2_acts = self.dstc2_acts_sys
        self.NActions = len(self.dstc2_acts)  # system acts without parameters
        self.NActions += len(self.system_requestable_slots)  # system request with certain slots
        self.NActions += len(self.requestable_slots)  # system inform with certain slot

        self.domain = Domain(self.dstc2_acts_sys,self.dstc2_acts_usr,self.system_requestable_slots,self.requestable_slots)

        self.agent:PolicyAgent = PolicyAgent(STATE_DIM, self.NActions)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-2)



    def initialize(self, **kwargs):
        """
        Initialize internal parameters

        :return: Nothing
        """

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

        if 'agent_role' in kwargs:
            self.agent_role = kwargs['agent_role']

    def restart(self, args):
        pass

    def next_action(self, state:SlotFillingDialogueState):

        # if (self.is_training and random.random() < self.epsilon):
        if (self.is_training):
            sys_acts = self.warmup_policy.next_action(state)
        else:
            state_enc = self.encode_state(state)
            action,_ = self.agent.step(numpy.array(state_enc,dtype=numpy.int64))
            sys_acts = self.decode_action(action)

        return sys_acts

    def encode_state(self, state:SlotFillingDialogueState):
        return encode_state(state, self.domain)

    def encode_action(self, actions, system=True):
        return encode_action(actions,system,self.domain)

    def decode_action(self, action_enc):
        return decode_action(action_enc,self.domain)

    @staticmethod
    def _calc_returns(exp, gamma,eps=numpy.finfo(numpy.float32).eps.item()):
        returns = []
        R = 0
        for action, log_prob, r in reversed(exp):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def train(self, dialogues):
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

        print('loss: %0.2f'%numpy.mean(losses))


        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.logger.debug('Q-Learning factors: [alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

    def save(self, path=None):
        pass

    def load(self, path=None):
        pass