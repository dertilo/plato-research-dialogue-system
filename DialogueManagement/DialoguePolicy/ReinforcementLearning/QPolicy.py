"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

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

"""
Q_Policy implements a simple Q-Learning dialogue policy.
"""

# create logger
module_logger = logging.getLogger(__name__)


class QPolicy(DialoguePolicy.DialoguePolicy):
    def __init__(self, ontology, database, agent_id=0, agent_role='system',
                 domain=None, alpha=0.95, epsilon=0.95,
                 gamma=0.15, alpha_decay=0.995, epsilon_decay=0.995, print_level='debug', epsilon_min=0.05):
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

        self.logger = logging.getLogger(__name__)
        self.warmup_mode = True
        self.print_level = print_level

        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
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

        self.database = None
        if isinstance(database, DataBase):
            self.database = database
        else:
            raise ValueError('WoLF PHC DialoguePolicy: Unacceptable database '
                             'type %s ' % database)

        self.Q = {}
        self.Q_info = {}  # tracks information about training of Q

        self.pp = pprint.PrettyPrinter(width=160)  # For debug!

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            # Put your system expert policy here
            self.warmup_policy = \
                HandcraftedPolicy.HandcraftedPolicy(self.ontology)

        elif self.agent_role == 'user':
            usim_args = \
                dict(
                    zip(['ontology', 'database'],
                        [self.ontology, self.database]))
            # Put your user expert policy here
            self.warmup_simulator = AgendaBasedUS(usim_args)

        # Extract lists of slots that are frequently used
        self.informable_slots = \
            deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = \
            deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        self.dstc2_acts = None

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions
            self.dstc2_acts = ['repeat', 'canthelp', 'affirm', 'negate',
                               'deny', 'ack', 'thankyou', 'bye',
                               'reqmore', 'hello', 'welcomemsg', 'expl-conf',
                               'select', 'offer', 'reqalts',
                               'confirm-domain', 'confirm']

        else:
            # Try to identify number of state features
            if domain in ['SlotFilling', 'CamRest']:

                # Plato does not use action masks (rules to define which
                # actions are valid from each state) and so training can
                # be harder. This becomes easier if we have a smaller
                # action set.

                # Sub-case for CamRest
                if domain == 'CamRest':
                    # Does not include inform and request that are modelled
                    # together with their arguments
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

                    if self.agent_role == 'system':
                        self.dstc2_acts = self.dstc2_acts_sys
                        self.NActions = len(self.dstc2_acts)  # system acts without parameters
                        self.NActions += len(self.system_requestable_slots)  # system request with certain slots
                        self.NActions += len(self.requestable_slots)  # system inform with certain slot

                    elif self.agent_role == 'user':
                        self.dstc2_acts = self.dstc2_acts_usr
                        self.NActions = len(self.dstc2_acts)  # user acts without parameters
                        self.NActions += len(self.requestable_slots)  # user request with certain slot
                        self.NActions += len(self.system_requestable_slots)  # user inform with certain slot
                    else:
                        self.logger.warning('Unknown agent role: "{}"'.format(self.agent_role))


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
        """
        Nothing to do here.

        :return:
        """

        pass

    def next_action(self, state):
        """
        Consults the policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        state_enc = self.encode_state(state)

        if state_enc not in self.Q or (self.is_training and
                                       random.random() < self.epsilon):

            threshold = 1.0 if self.warmup_mode else 0.5
            if random.random() < threshold:
                # During exploration we may want to follow another policy,
                # e.g. an expert policy.

                if self.print_level in ['debug']:
                    print('---: Selecting warmup action.')

                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)
                else:
                    self.warmup_simulator.receive_input(
                        state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()

            else:
                # Return a random action
                if self.print_level in ['debug']:
                    print('---: Selecting random action')
                return self.decode_action(
                    random.choice(
                        range(0, self.NActions)),
                    self.agent_role == 'system')

        if self.IS_GREEDY_POLICY:
            # Return action with maximum Q value from the given state
            sys_acts = self.decode_action(max(self.Q[state_enc],
                                              key=self.Q[state_enc].get),
                                          self.agent_role == 'system')
        else:
            sys_acts = self.decode_action(
                random.choices(
                    range(0, self.NActions),
                    self.Q[state_enc])[0],
                self.agent_role == 'system')

        return sys_acts

    def encode_state(self, state):
        """
        Encodes the dialogue state into an index used to address the Q matrix.

        :param state: the state to encode
        :return: int - a unique state ID
        """

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
                return [int(b) for b in
                     format(int(round(state.db_matches_ratio, 2) * 100), '07b')]
            else:
                # If the number is negative (should not happen in general) there
                # will be a minus sign
                return [int(b) for b in
                     format(int(round(state.db_matches_ratio, 2) * 100),
                            '07b')[1:]]

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
                return [int(b) for b in format(integer, '04b')]
            else:
                return [0, 0, 0, 0]
        # --------------------------------------------------------------------------
        temp = []

        temp += [int(b) for b in format(state.turn, '06b')]

        for value in state.slots_filled.values():
            # This contains the requested slot
            temp.append(1) if value else temp.append(0)

        for slot in self.ontology.ontology['requestable']:
            temp.append(1) if slot == state.requested_slot else temp.append(0)

        temp.append(int(state.is_terminal_state))

        temp += encode_item_in_focus(state)
        temp += encode_db_matches_ratio(state)
        temp.append(1) if state.system_made_offer else temp.append(0)
        temp += encode_user_acts(state)
        temp += encode_last_sys_acts(state)

        state_enc = sum([2**k for k,b in enumerate(reversed(temp)) if b==1])
        return state_enc



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

    def decode_action(self, action_enc, system=True):
        """
        Decode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be decoding another agent's action
        (e.g. a system decoding the previous user act).

        :param action_enc: action encoding to be decoded
        :param system: whether the role whose action we are decoding is a
                       'system'
        :return: the decoded action
        """

        if system:  # decode for system
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

        else:  # decode for user
            if action_enc < len(self.dstc2_acts_usr):
                return [DialogueAct(self.dstc2_acts_usr[action_enc], [])]

            if action_enc < len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots):
                return [DialogueAct(
                    'request',
                    [DialogueActItem(
                        self.requestable_slots[
                            action_enc - len(self.dstc2_acts_usr)],
                        Operator.EQ,
                        '')])]

            if action_enc < len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots):
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.system_requestable_slots[
                            action_enc - len(self.dstc2_acts_usr) -
                            len(self.requestable_slots)], Operator.EQ, '')])]

    def train(self, dialogues):
        """
        Train the model using Q-learning.

        :param dialogues: a list dialogues, which is a list of dialogue turns
                          (state, action, reward triplets).
        :return:
        """
        self.warmup_mode = True
        self.logger.info('Train Q with {} dialogues'.format(len(dialogues)))
        for k,dialogue in enumerate(dialogues):
            # if k>50:
            #     self.warmup_mode = False
            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            for turn in dialogue:
                state_enc = self.encode_state(turn['state'])
                new_state_enc = self.encode_state(turn['new_state'])
                action_enc = self.encode_action(turn['action'])

                if action_enc < 0:
                    continue

                if state_enc not in self.Q:
                    self.Q[state_enc] = {}

                if action_enc not in self.Q[state_enc]:
                    self.Q[state_enc][action_enc] = 0

                max_q = 0
                if new_state_enc in self.Q:
                    max_q = max(self.Q[new_state_enc].values())

                new_q = self.alpha * (turn['reward'] +
                                      self.gamma * max_q -
                                      self.Q[state_enc][action_enc])

                # self.logger.debug('Old q: {}, New q: {}, Diff: {}'.format(self.Q[state_enc][action_enc], new_q,
                #                                                          new_q - self.Q[state_enc][action_enc]))

                self.Q[state_enc][action_enc] += new_q

                # count update of Q
                if state_enc not in self.Q_info:
                    self.Q_info[state_enc] = {}
                if action_enc not in self.Q_info[state_enc]:
                    self.Q_info[state_enc][action_enc] = 0
                self.Q_info[state_enc][action_enc] += 1

        # Decay learning rate
        if self.alpha > 0.001:
            self.alpha *= self.alpha_decay

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.logger.debug('Q-Learning factors: [alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

    def decay_epsilon(self):
        """
        Decays epsilon (exploration rate) by epsilon decay.

         Decays epsilon (exploration rate) by epsilon decay.
         If epsilon is already less or equal compared to epsilon_min,
         the call of this method has no effect.

        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path=None):
        """
        Save the Q learning policy model

        :param path: the path to save the model to
        :return: nothing
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'Models/Policies/q_policy.pkl'
            self.logger.warning('No policy file name provided. Using default: {0}'.format(path))

        obj = {'Q': self.Q,
               'a': self.alpha,
               'e': self.epsilon,
               'e_decay': self.epsilon_decay,
               'e_min': self.epsilon_min,
               'g': self.gamma,
               'i': self.Q_info}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

    def load(self, path=None):
        """
        Loads the Q learning policy model

        :param path: the path to load the model from
        :return: nothing
        """

        self.logger.info('Load policy model.'.format(path))

        if not path:
            self.logger.warning('No policy loaded.')
            return

        if isinstance(path, str):
            if os.path.isfile(path):
                with open(path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'Q' in obj:
                        self.Q = obj['Q']
                        self.logger.debug('Number of states in Q: {}'.format(len(self.Q)))
                        actions = set()
                        for k, v in self.Q.items():
                            actions.update(list(v.keys()))
                        self.logger.debug('Q contains {} distinct actions: {}'.format(len(actions), actions))
                    if 'a' in obj:
                        self.alpha = obj['a']
                        self.logger.debug('Alpha from loaded policy: {}'.format(obj['a']))
                    if 'e' in obj:
                        self.epsilon = obj['e']
                        self.logger.debug('Epsilon from loaded policy: {}'.format(obj['e']))
                    if 'e_decay' in obj:
                        self.epsilon_decay = obj['e_decay']
                    if 'e_min' in obj:
                        self.epsilon_min = obj['e_min']
                    if 'g' in obj:
                        self.gamma = obj['g']
                        self.logger.debug('Gamma from loaded policy: {}'.format(obj['g']))
                    if 'i' in obj:
                        self.Q_info = obj['i']

                    self.logger.info('Q DialoguePolicy loaded from {0}.'.format(path))

            else:
                self.logger.warning('Warning! Q DialoguePolicy file {} not found'.format(path))
        else:
            self.logger.warning('Unacceptable value for Q policy file name: {} '.format(path))
