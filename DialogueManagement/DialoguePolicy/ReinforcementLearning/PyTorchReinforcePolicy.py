import numpy

from Dialogue.State import SlotFillingDialogueState
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
    def __init__(self, obs_dim, num_actions,hidden_dim = 32) -> None:
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


STATE_DIM = 45

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

        def encode_item_in_focus(state):
            # If the agent is a system, then this shows what the top db result is.
            # If the agent is a user, then this shows what information the
            # system has provided
            out = []
            if state.item_in_focus:
                for slot in self.ontology.ontology['requestable']:
                    if slot in state.item_in_focus and state.item_in_focus[slot]:
                        out.append(1)
                    else:
                        out.append(0)
            else:
                out = [0] * len(self.ontology.ontology['requestable'])
            return out

        def encode_db_matches_ratio(state):
            if state.db_matches_ratio >= 0:
                out = [int(b) for b in
                     format(int(round(state.db_matches_ratio, 2) * 100), '07b')]
            else:
                # If the number is negative (should not happen in general) there
                # will be a minus sign
                out = [int(b) for b in
                     format(int(round(state.db_matches_ratio, 2) * 100),
                            '07b')[1:]]
            assert len(out)==7
            return out

        def encode_user_acts(state):
            if state.user_acts:
                return [int(b) for b in
                     format(self.encode_action(state.user_acts, False), '05b')]
            else:
                return [0, 0, 0, 0, 0]

        def encode_last_sys_acts(state):
            if state.last_sys_acts:
                integer = self.encode_action([state.last_sys_acts[0]])
                # assert integer<16 # TODO(tilo):
                out = [int(b) for b in format(integer, '05b')]
            else:
                out = [0, 0, 0, 0, 0]
            assert len(out)==5
            return out

        def encode_slots_filled_values(state):
            out = []
            for value in state.slots_filled.values():
                # This contains the requested slot
                out.append(1) if value else out.append(0)
            assert len(out)==6
            return out

        # --------------------------------------------------------------------------
        temp = []

        temp += [int(b) for b in format(state.turn, '06b')]

        temp += encode_slots_filled_values(state)

        for slot in self.ontology.ontology['requestable']:
            temp.append(1) if slot == state.requested_slot else temp.append(0)

        temp.append(int(state.is_terminal_state))

        temp += encode_item_in_focus(state)
        temp += encode_db_matches_ratio(state)
        temp.append(1) if state.system_made_offer else temp.append(0)
        temp += encode_user_acts(state)
        temp += encode_last_sys_acts(state)

        # state_enc = sum([2**k for k,b in enumerate(reversed(temp)) if b==1])
        assert len(temp)==STATE_DIM
        return temp


    def encode_action(self, actions, system=True):
        """
        Encode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be encoding another agent's action
        (e.g. a system encoding the previous user act).

        :param actions: actions to be encoded
        :param system: whether the role whose action we are encoding is a
                       'system'
        :return: the encoded action
        """

        # TODO: Handle multiple actions
        # TODO: Action encoding in a principled way
        if not actions:
            print('WARNING: Supervised DialoguePolicy action encoding called '
                  'with empty actions list (returning -1).')
            return -1

        action = actions[0]
        intent = action.intent

        slot = None
        if action.params and action.params[0].slot:
            slot = action.params[0].slot

        enc = None
        if system:  # encode for system
            if self.dstc2_acts_sys and intent in self.dstc2_acts_sys:
                enc = self.dstc2_acts_sys.index(action.intent)

            elif slot:
                if intent == 'request' and slot in self.system_requestable_slots:
                    enc = len(self.dstc2_acts_sys) + self.system_requestable_slots.index(
                        slot)

                elif intent == 'inform' and slot in self.requestable_slots:
                    enc = len(self.dstc2_acts_sys) + len(
                        self.system_requestable_slots) + self.requestable_slots.index(slot)
        else:
            if self.dstc2_acts_usr and intent in self.dstc2_acts_usr:
                enc =  self.dstc2_acts_usr.index(action.intent)

            elif slot:
                if intent == 'request' and slot in self.requestable_slots:
                    enc = len(self.dstc2_acts_usr) + \
                           self.requestable_slots.index(slot)

                elif action.intent == 'inform' and slot in self.system_requestable_slots:
                    enc = len(self.dstc2_acts_usr) + \
                           len(self.requestable_slots) + \
                           self.system_requestable_slots.index(slot)
        if enc is None:
            # Unable to encode action
            print('Q-Learning ({0}) policy action encoder warning: Selecting '
                  'default action (unable to encode: {1})!'.format(self.agent_role, action))
            enc = -1

        return enc

    def decode_action(self, action_enc):
        """
        Decode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be decoding another agent's action
        (e.g. a system decoding the previous user act).

        :param action_enc: action encoding to be decoded
        :param system: whether the role whose action we are decoding is a
                       'system'
        :return: the decoded action
        """
        if action_enc < len(self.dstc2_acts_sys):
            return [DialogueAct(self.dstc2_acts_sys[action_enc], [])]

        if action_enc < len(self.dstc2_acts_sys) + \
                len(self.system_requestable_slots):
            return [DialogueAct(
                'request',
                [DialogueActItem(
                    self.system_requestable_slots[
                        action_enc - len(self.dstc2_acts_sys)],
                    Operator.EQ,
                    '')])]

        if action_enc < len(self.dstc2_acts_sys) + \
                len(self.system_requestable_slots) + \
                len(self.requestable_slots):
            index = \
                action_enc - len(self.dstc2_acts_sys) - \
                len(self.system_requestable_slots)
            return [DialogueAct(
                'inform',
                [DialogueActItem(
                    self.requestable_slots[index], Operator.EQ, '')])]

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