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
from Domain.Ontology import Ontology
from Domain.DataBase import DataBase
from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS

from copy import deepcopy

import pickle
import random
import pprint
import os.path
import numpy as np
import logging
from typing import List, Dict

from DialogueManagement.DialoguePolicy.dialogue_common import setup_domain, \
    create_random_dialog_act, action_to_string, state_to_json

"""
WoLF_PHC_Policy implements a Win or Lose Fast DialoguePolicy Hill Climbing 
dialogue policy learning algorithm, designed for multi-agent systems.
"""


class WoLFPHCPolicy(DialoguePolicy.DialoguePolicy):
    def __init__(self, ontology, database, agent_id=0, agent_role='system',
                 alpha=0.25, gamma=0.95, epsilon=0.25,
                 alpha_decay=0.9995, epsilon_decay=0.995, epsilon_min=0.05,
                 warm_up_mode=False, **kwargs):
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
        self.warm_up_mode = False
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.IS_GREEDY_POLICY = False

        # TODO: Put these as arguments in the config
        self.d_win = 0.0025
        self.d_lose = 0.01

        self.is_training = False

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
        self.pi = {}
        self.mean_pi = {}
        self.state_counter = {}

        self.pp = pprint.PrettyPrinter(width=160)     # For debug!

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            # Put your system expert policy here
            self.warmup_policy = \
                HandcraftedPolicy.HandcraftedPolicy(self.ontology)

        elif self.agent_role == 'user':
            usim_args = dict(
                zip(['ontology', 'database'], [self.ontology, self.database]))
            # Put your user expert policy here
            self.warmup_simulator = AgendaBasedUS(usim_args)


        # Plato does not use action masks (rules to define which
        # actions are valid from each state) and so training can
        # be harder. This becomes easier if we have a smaller
        # action set.

        # Extract lists of slots that are frequently used
        self.informable_slots = deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = deepcopy(self.ontology.ontology['system_requestable'])

        self.statistics = {'supervised_turns': 0, 'total_turns': 0}

        self.hash2actions = {}

        self.domain = setup_domain(self.ontology)
        self.NActions = self.domain.NActions

    def initialize(self, **kwargs):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        if 'is_training' in kwargs:
            self.is_training = bool(kwargs['is_training'])

            if 'learning_rate' in kwargs:
                self.alpha = float(kwargs['learning_rate'])

            if 'learning_decay_rate' in kwargs:
                self.alpha_decay = float(kwargs['learning_decay_rate'])

            if 'exploration_rate' in kwargs:
                self.epsilon = float(kwargs['exploration_rate'])

            if 'exploration_decay_rate' in kwargs:
                self.epsilon_decay = float(kwargs['exploration_decay_rate'])

            if 'gamma' in kwargs:
                self.gamma = float(kwargs['gamma'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in kwargs:
                    self.warmup_simulator.initialize({kwargs['goal']})
                else:
                    self.logger.warning('WARNING ! No goal provided for Supervised policy '
                                        'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

    def restart(self, args):
        """
        Re-initialize relevant parameters / variables at the beginning of each
        dialogue.

        :return: nothing
        """

        if self.agent_role == 'user' and self.warmup_simulator:
            if 'goal' in args:
                self.warmup_simulator.initialize(args)
            else:
                self.logger.warning('WARNING! No goal provided for Supervised policy user '
                                    'simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        # state_enc = self.encode_state(state)
        state_enc = state_to_json(state)
        self.statistics['total_turns'] += 1

        if state_enc not in self.pi or \
                (self.is_training and random.random() < self.epsilon):
            if not self.is_training:
                if not self.pi:
                    self.logger.warning(f'\nWARNING! WoLF-PHC pi is empty '
                                        f'({self.agent_role}). Did you load the correct '
                                        f'file?\n')
                else:
                    self.logger.warning(f'\nWARNING! WoLF-PHC state not found in policy '
                                        f'pi ({self.agent_role}).\n')
            threshold = 1.0 if self.warm_up_mode else 0.5
            if self.is_training and random.random() < threshold:
                # use warm up / hand crafted only in training
                self.logger.debug('--- {0}: Selecting warmup action.'
                                  .format(self.agent_role))
                self.statistics['supervised_turns'] += 1

                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)

                else:
                    self.warmup_simulator.receive_input(
                        state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()
            else:
                self.logger.debug('--- {0}: Selecting random action.'.format(self.agent_role))
                sys_acts = create_random_dialog_act(self.domain, is_system=True)
                return sys_acts

        if self.IS_GREEDY_POLICY:
            # Get greedy action
            # Do not consider 'UNK' or an empty action
            state_actions = {}
            for k, v in self.pi[state_enc].items():
                if k and len(k) > 0:
                    state_actions[k] = v

            if len(state_actions) < 1:
                self.logger.warning('--- {0}: Warning! No maximum value identified for '
                                    'policy. Selecting random action.'
                                    .format(self.agent_role))

                sys_acts = create_random_dialog_act(self.domain, is_system=True)
            else:

                # find all actions with same max_value
                max_value = max(state_actions.values())
                max_actions = [k for k, v in state_actions.items() if v == max_value]

                # break ties randomly
                action = random.choice(max_actions)
                sys_acts = self.decode_action(action, system=True)

        else:
            # Sample next action
            action_from_pi = random.choices(list(self.pi[state_enc].keys()), list(self.pi[state_enc].values()))[0]
            sys_acts = self.decode_action(action_from_pi, self.agent_role == 'system')

        assert sys_acts is not None
        return sys_acts

    def encode_action(self, acts:List[DialogueAct], system=True) -> str:
        """
        Encode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be encoding another agent's action
        (e.g. a system encoding the previous user act).

        :param actions: actions to be encoded
        :param system: whether the role whose action we are encoding is a
                       'system'
        :return: the encoded action
        """

        acts_copy = [deepcopy(x) for x in acts]
        for act in acts_copy:
            if act.params:
                for item in act.params:
                    if item.slot and item.value:
                        item.value = None

        s = action_to_string(acts_copy, system)
        # enc = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 32)
        self.hash2actions[s] = acts_copy
        return s

    def decode_action(self, action_enc: str, system=True) -> List[DialogueAct]:
        """
        Decode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be decoding another agent's action
        (e.g. a system decoding the previous user act).

        :param action_enc: action encoding to be decoded
        :param system: whether the role whose action we are decoding is a
                       'system'
        :return: the decoded action
        """

        return self.hash2actions[action_enc]

    def train(self, dialogues):
        """
        Train the model using WoLF-PHC.

        :param dialogues: a list dialogues, which is a list of dialogue turns
                         (state, action, reward triplets).
        :return:
        """

        if not self.is_training:
            return

        for dialogue in dialogues:
            if len(dialogue) > 1:
                dialogue[-2]['reward'] = dialogue[-1]['reward']

            for turn in dialogue:
                state_enc = state_to_json(turn['state'])
                new_state_enc = state_to_json(turn['new_state'])
                action_enc = \
                    self.encode_action(
                        turn['action'],
                        self.agent_role == 'system')

                # Skip unrecognised actions
                if not action_enc or turn['action'][0].intent == 'bye':
                    continue

                if state_enc not in self.Q:
                    self.Q[state_enc] = {}

                if action_enc not in self.Q[state_enc]:
                    self.Q[state_enc][action_enc] = 0

                if new_state_enc not in self.Q:
                    self.Q[new_state_enc] = {}

                # add the current action to new_state to have have at least one value for the new state when updating Q
                # if action_enc not in self.Q[new_state_enc]:
                #    self.Q[new_state_enc][action_enc] = 0

                if state_enc not in self.pi:
                    # self.pi[state_enc] = \
                    #    [float(1/self.NActions)] * self.NActions
                    self.pi[state_enc] = {}

                if action_enc not in self.pi[state_enc]:
                    self.pi[state_enc][action_enc] = float(1/self.NActions)

                if state_enc not in self.mean_pi:
                    #self.mean_pi[state_enc] = \
                    #    [float(1/self.NActions)] * self.NActions
                    self.mean_pi[state_enc] = {}

                if action_enc not in self.mean_pi[state_enc]:
                    self.mean_pi[state_enc][action_enc] = float(1/self.NActions)

                if state_enc not in self.state_counter:
                    self.state_counter[state_enc] = 1
                else:
                    self.state_counter[state_enc] += 1

                # Update Q
                max_new_state = max(self.Q[new_state_enc].values()) if len(self.Q[new_state_enc]) > 0 else 0
                self.Q[state_enc][action_enc] = \
                    ((1 - self.alpha) * self.Q[state_enc][action_enc]) + \
                    self.alpha * (
                            turn['reward'] +
                            (self.gamma * max_new_state))

                # Update mean policy estimate
                # for a in range(self.NActions):
                for a in self.mean_pi[state_enc].keys():
                    self.mean_pi[state_enc][a] = \
                        self.mean_pi[state_enc][a] + \
                        ((1.0 / self.state_counter[state_enc]) *
                         (self.pi[state_enc][a] - self.mean_pi[state_enc][a]))

                # Determine delta
                sum_policy = 0.0
                sum_mean_policy = 0.0

                # for a in range(self.NActions):
                for a in self.Q[state_enc].keys():
                    sum_policy = sum_policy + (self.pi[state_enc][a] *
                                               self.Q[state_enc][a])
                    sum_mean_policy = \
                        sum_mean_policy + \
                        (self.mean_pi[state_enc][a] * self.Q[state_enc][a])

                if sum_policy > sum_mean_policy:
                    delta = self.d_win
                else:
                    delta = self.d_lose

                # Update policy estimate
                max_action_Q = max(self.Q[state_enc], key=self.Q[state_enc].get)

                d_plus = delta
                d_minus = ((-1.0) * d_plus) / (self.NActions - 1.0)

                # for a in range(self.NActions):
                for a in self.Q[state_enc].keys():
                    if a == max_action_Q:
                        self.pi[state_enc][a] = \
                            min(1.0, self.pi[state_enc][a] + d_plus)
                    else:
                        self.pi[state_enc][a] = \
                            max(0.0, self.pi[state_enc][a] + d_minus)

                # Constrain pi to a legal probability distribution
                # use max, as NActions is rather an estimate...
                num_unseen_actions = max(self.NActions - len(self.pi[state_enc]), 0)
                sum_unseen_actions = num_unseen_actions * float(1/self.NActions)
                sum_pi = sum(self.pi[state_enc].values()) + sum_unseen_actions
                # for a in range(self.NActions):
                for a in self.pi[state_enc].keys():
                    self.pi[state_enc][a] /= sum_pi

        # Decay learning rate after each episode
        if self.alpha > 0.001:
            self.alpha *= self.alpha_decay

        # Decay exploration rate after each episode
        self.decay_epsilon()

        self.logger.info('[alpha: {0}, epsilon: {1}]'.format(self.alpha, self.epsilon))

    def decay_epsilon(self):
        """
        Decays epsilon (exploration rate) by epsilon decay.

         Decays epsilon (exploration rate) by epsilon decay.
         If epsilon is already less or equal compared to epsilon_min,
         the call of this method has no effect.

        :return:
        """
        if self.epsilon > self.epsilon_min and not self.warm_up_mode:
            self.epsilon *= self.epsilon_decay


    def save(self, path=None):
        """
        Saves the policy model to the path provided

        :param path: path to save the model to
        :return:
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'Models/Policies/wolf_phc_policy.pkl'
            self.logger.warning('No policy file name provided. Using default: {0}'.format(path))

        obj = {'Q': self.Q,
               'pi': self.pi,
               'mean_pi': self.mean_pi,
               'state_counter': self.state_counter,
               'a': self.alpha,
               'e': self.epsilon,
               'e_decay': self.epsilon_decay,
               'e_min': self.epsilon_min,
               'g': self.gamma,
               'hash2actions': self.hash2actions}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        if self.statistics['total_turns'] > 0:
            self.logger.debug('{0} WoLF PHC DialoguePolicy supervision ratio: {1}'
                              .format(self.agent_role,
                                      float(
                                          self.statistics['supervised_turns'] /
                                          self.statistics['total_turns'])))

        self.logger.debug(f'{self.agent_role} WoLF PHC DialoguePolicy state space '
                          f'size: {len(self.pi)}')

    def load(self, path=None):
        """
        Load the policy model from the path provided

        :param path: path to load the model from
        :return:
        """

        if not path:
            self.logger.info('No policy loaded.')
            return

        if isinstance(path, str):
            if os.path.isfile(path):
                with open(path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'Q' in obj:
                        self.Q = obj['Q']
                    if 'pi' in obj:
                        self.pi = obj['pi']
                    if 'mean_pi' in obj:
                        self.mean_pi = obj['mean_pi']
                    if 'state_counter' in obj:
                        self.state_counter = obj['state_counter']
                    if 'a' in obj:
                        self.alpha = obj['a']
                    if 'e' in obj:
                        self.epsilon = obj['e']
                    if 'e_decay' in obj:
                        self.epsilon_decay = obj['e_decay']
                    if 'e_min' in obj:
                        self.epsilon_min = obj['e_min']
                    if 'g' in obj:
                        self.gamma = obj['g']
                    if 'hash2actions' in obj:
                        self.hash2actions = obj['hash2actions']

                    self.logger.info('WoLF-PHC DialoguePolicy loaded from {0}.'.format(path))

            else:
                self.logger.warning('Warning! WoLF-PHC DialoguePolicy file %s not found' % path)
        else:
            self.logger.warning('Warning! Unacceptable value for WoLF-PHC policy file name: %s ' % path)
