"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List

from DialogueManagement.DialoguePolicy.dialogue_common import setup_domain, \
    create_random_dialog_act, action_to_string, state_to_json

__author__ = "Alexandros Papangelis"

from .. import DialoguePolicy, HandcraftedPolicy
from Dialogue.Action import DialogueAct
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from copy import deepcopy

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
                 gamma=0.15, alpha_decay=0.995, epsilon_decay=0.995, print_level='debug', epsilon_min=0.05,
                 warm_up_mode=False, **kwargs
                 ):
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
        self.warm_up_mode = warm_up_mode
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

        self.warmup_policy = \
            HandcraftedPolicy.HandcraftedPolicy(self.ontology)

        self.domain = setup_domain(self.ontology)
        self.NActions = self.domain.NActions

        self.hash2actions = {}
        self.counter = {'warmup':0,'learned':0,'random':0}

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

        state_enc = state_to_json(state)
        assert self.IS_GREEDY_POLICY
        if self.is_training and (state_enc not in self.Q or random.random() < self.epsilon):

            threshold = 1.0 if self.warm_up_mode else 0.5
            if random.random() < threshold:
                # During exploration we may want to follow another policy,
                # e.g. an expert policy.

                if self.print_level in ['debug']:
                    print('---: Selecting warmup action.')

                if self.agent_role == 'system':
                    sys_acts = self.warmup_policy.next_action(state)
                else:
                    self.warmup_simulator.receive_input(
                        state.user_acts, state.user_goal)
                    sys_acts = self.warmup_simulator.respond()
                self.counter['warmup']+=1
            else:
                # Return a random action
                if self.print_level in ['debug']:
                    print('---: Selecting random action')
                sys_acts = create_random_dialog_act(self.domain,is_system=True)
                self.counter['random'] += 1

        elif self.IS_GREEDY_POLICY and state_enc in self.Q:
            # Return action with maximum Q value from the given state
            sys_acts = self.decode_action(max(self.Q[state_enc],
                                              key=self.Q[state_enc].get),
                                          )
            self.counter['learned'] += 1
        else:
            # Return a random action
            if self.print_level in ['debug']:
                print('---: Selecting random action')
            sys_acts = create_random_dialog_act(self.domain, is_system=True)
            self.counter['random'] += 1
        assert sys_acts is not None
        return sys_acts


    def encode_action(self, acts:List[DialogueAct], system=True)->str:
        acts_copy = [deepcopy(x) for x in acts]
        for act in acts_copy:
            if act.params:
                for item in act.params:
                    if item.slot and item.value:
                        item.value = None

        s = action_to_string(acts_copy, system)
        # enc = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 32)
        self.hash2actions[s]=acts_copy
        return s

    def decode_action(self, action_enc):
        return self.hash2actions[action_enc]

    def train(self, dialogues):
        """
        Train the model using Q-learning.

        :param dialogues: a list dialogues, which is a list of dialogue turns
                          (state, action, reward triplets).
        :return:
        """
        self.logger.info('Train Q with {} dialogues'.format(len(dialogues)))
        for k,dialogue in enumerate(dialogues):
            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            for turn in dialogue:
                state_enc = state_to_json(turn['state'])
                new_state_enc = state_to_json(turn['new_state'])
                action_enc = self.encode_action(turn['action'])

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
        if self.alpha > 0.001:#TODO(tilo):why?
            self.alpha *= self.alpha_decay

        # Decay exploration rate
        if self.epsilon > self.epsilon_min and not self.warm_up_mode:
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
               'i': self.Q_info,
               'hash2actions':self.hash2actions
               }

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
                    if 'hash2actions' in obj:
                        self.hash2actions = obj['hash2actions']

                    self.logger.info('Q DialoguePolicy loaded from {0}.'.format(path))

            else:
                self.logger.warning('Warning! Q DialoguePolicy file {} not found'.format(path))
        else:
            self.logger.warning('Unacceptable value for Q policy file name: {} '.format(path))
