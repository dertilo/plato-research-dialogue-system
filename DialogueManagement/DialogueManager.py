"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_a2c_policy import \
    PyTorchA2CPolicy
from DialogueManagement.DialoguePolicy.ReinforcementLearning.pytorch_reinforce_policy import \
    PyTorchReinforcePolicy

__author__ = "Alexandros Papangelis"

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from DialogueManagement.dialogue_management import build_offer, build_inform, \
    build_explicit_confirm, build_offer_from_inform
from DialogueStateTracker.DialogueStateTracker import DummyStateTracker
# from DialogueStateTracker.CamRestLudwigDST import CamRestLudwigDST

from DialogueManagement.DialoguePolicy.HandcraftedPolicy import \
    HandcraftedPolicy
from DialogueManagement.DialoguePolicy.CalculatedPolicy import \
    CalculatedPolicy
from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import \
    QPolicy
from DialogueManagement.DialoguePolicy.ReinforcementLearning.MinimaxQPolicy \
    import MinimaxQPolicy
from DialogueManagement.DialoguePolicy.ReinforcementLearning.WoLFPHCPolicy \
    import WoLFPHCPolicy
# from DialogueManagement.DialoguePolicy.DeepLearning.SupervisedPolicy import \
#     SupervisedPolicy
from DialogueManagement.DialoguePolicy.DeepLearning.ReinforcePolicy import \
    ReinforcePolicy

from Domain.Ontology import Ontology
from Domain.DataBase import DataBase, SQLDataBase, JSONDataBase

from copy import deepcopy

from ConversationalAgent.ConversationalModule import ConversationalModule

import random
import math

"""
The DialogueManager consists of a DialogueStateTracker and a DialoguePolicy. 
It handles the decision-making part of the Conversational Agent. 
Given new input (a list of DialogueActs) it will ensure that the state is 
updated properly and will output a list of DialogueActs in response, after 
querying its DialoguePolicy.
"""


class DialogueManager(ConversationalModule):
    def __init__(self, args):
        """
        Parses the arguments in the dictionary and initializes the appropriate
        models for Dialogue State Tracking and Dialogue Policy.

        :param args: the configuration file parsed into a dictionary
        """

        if 'settings' not in args:
            raise AttributeError(
                'DialogueManager: Please provide settings (config)!')
        if 'ontology' not in args:
            raise AttributeError('DialogueManager: Please provide ontology!')
        if 'database' not in args:
            raise AttributeError('DialogueManager: Please provide database!')
        if 'domain' not in args:
            raise AttributeError('DialogueManager: Please provide domain!')

        settings = args['settings']
        ontology = args['ontology']
        database = args['database']
        domain = args['domain']

        agent_id = 0
        if 'agent_id' in args:
            agent_id = int(args['agent_id'])

        agent_role = 'system'
        if 'agent_role' in args:
            agent_role = args['agent_role']

        self.settings = settings
        self.print_level = 'debug'
        if 'GENERAL' in settings and 'print_level' in settings['GENERAL']:
            self.print_level = settings['GENERAL']['print_level']

        self.TRAIN_DST = False
        self.TRAIN_POLICY = False

        self.MAX_DB_RESULTS = 10

        self.DSTracker = None
        self.policy = None
        self.policy_path = None
        self.ontology = None
        self.database = None
        self.domain = None

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.dialogue_counter = 0
        self.CALCULATE_SLOT_ENTROPIES = True

        # True: use an inform act to answer the request for the name of an item
        # False: use an offer act to answer the request for the name of an item
        self.inform_requested_name = True

        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        if isinstance(database, DataBase):
            self.database = database

        elif isinstance(database, str):
            if database[-3:] == '.db':
                self.database = SQLDataBase(database)
            elif database[-5:] == '.json':
                self.database = JSONDataBase(database)
            else:
                raise ValueError('Unacceptable database type %s ' % database)

        else:
            raise ValueError('Unacceptable database type %s ' % database)
                
        if args and args['policy']:
            if 'domain' in self.settings['DIALOGUE']:
                self.domain = self.settings['DIALOGUE']['domain']
            else:
                raise ValueError(
                    'Domain is not specified in DIALOGUE at config.')

            if 'calculate_slot_entropies' in args:
                self.CALCULATE_SLOT_ENTROPIES = \
                    bool(args['calculate_slot_entropies'])

            self.init_policy(args)

        # DST Settings
        if 'DST' in args and args['DST']['dst']:
                if args['DST']['dst'] == 'CamRest':
                    if args['DST']['policy']['model_path'] and \
                            args['DST']['policy']['metadata_path']:
                        self.DSTracker = \
                            CamRestLudwigDST(
                                {'model_path': args[
                                    'DST']['policy']['model_path']})
                    else:
                        raise ValueError(
                            'Cannot find model_path or metadata_path in the '
                            'config for dialogue state tracker.')

        # Default to dummy DST
        if not self.DSTracker:
            dst_args = dict(
                zip(
                    ['ontology', 'database', 'domain'],
                    [self.ontology, self.database, domain]))
            self.DSTracker = DummyStateTracker(dst_args)

        self.load('')

    def init_policy(self, args):
        if not args or not args['policy']:
            # Early return
            return

        # collect all (potential) parameters
        # read float number parameters from policy section in config
        # Actually we not only read floats, but also translate between different namings of paramenters
        #  in config files and code.
        float_params_key_map = {'learning_rate':'alpha',
                   'discount_factor':'gamma',
                   'exploration_rate':'epsilon',
                   'learning_decay_rate':'alpha_decay',
                   'exploration_decay_rate':'epsilon_decay',
                   'min_exploration_rate':'epsilon_min'}
        policy_params = {float_params_key_map[k]:float(v) for k,v in args['policy'].items() if k in float_params_key_map}

        # Read parameters from policy section which have equal names in config files and code.
        policy_params.update({k:v for k,v in args['policy'].items() if k not in float_params_key_map})

        # initialize the policy (depending on the configured policy type)
        if args['policy']['type'] == 'handcrafted':
            self.policy = HandcraftedPolicy(self.ontology)

        elif args['policy']['type'] == 'q_learning':

            self.policy = \
                QPolicy(self.ontology,
                        self.database,
                        self.agent_id,
                        self.agent_role,
                        self.domain,
                        print_level=self.print_level,
                        **policy_params)

        elif args['policy']['type'] == 'pytorch_reinforce':

            self.policy = \
                PyTorchReinforcePolicy(self.ontology,
                        self.database,
                        self.agent_id,
                        self.agent_role,
                        self.domain,
                        print_level=self.print_level,
                        **policy_params)

        elif args['policy']['type'] == 'pytorch_a2c':

            self.policy = \
                PyTorchA2CPolicy(self.ontology,
                        self.database,
                        self.agent_id,
                        self.agent_role,
                        self.domain,
                        print_level=self.print_level,
                        **policy_params)

        elif args['policy']['type'] == 'minimax_q':

            self.policy = \
                MinimaxQPolicy(
                    self.ontology,
                    self.database,
                    self.agent_id,
                    self.agent_role,
                    **policy_params)

        elif args['policy']['type'] == 'wolf_phc':

            self.policy = \
                WoLFPHCPolicy(
                    self.ontology,
                    self.database,
                    self.agent_id,
                    self.agent_role,
                    **policy_params)

        elif args['policy']['type'] == 'reinforce':

            self.policy = \
                ReinforcePolicy(
                    self.ontology,
                    self.database,
                    self.agent_id,
                    self.agent_role,
                    self.domain,
                    **policy_params)

        elif args['policy']['type'] == 'calculated':
            self.policy = \
                CalculatedPolicy(
                    self.ontology,
                    self.database,
                    self.agent_id,
                    self.agent_role,
                    self.domain)

        elif args['policy']['type'] == 'supervised':
            self.policy = \
                SupervisedPolicy(
                    self.ontology,
                    self.database,
                    self.agent_id,
                    self.agent_role,
                    self.domain)

        elif args['policy']['type'] == 'ludwig':
            if args['policy']['policy_path']:
                print('DialogueManager: Instantiate your ludwig-based'
                      'policy here')
            else:
                raise ValueError(
                    'Cannot find policy_path in the config for dialogue '
                    'policy.')
        else:
            raise ValueError('DialogueManager: Unsupported policy type!'
                             .format(args['policy']['type']))

        if 'train' in args['policy']:
            self.TRAIN_POLICY = bool(args['policy']['train'])

        if 'policy_path' in args['policy']:
            self.policy_path = args['policy']['policy_path']

    def initialize(self, args):
        """
        Initialize the relevant structures and variables of the Dialogue
        Manager.

        :return: Nothing
        """

        self.DSTracker.initialize()
        if 'goal' not in args:
            self.policy.initialize(
                **{'is_training': self.TRAIN_POLICY,
                   'policy_path': self.policy_path,
                   'ontology': self.ontology})
        else:
            self.policy.initialize(
                **{'is_training': self.TRAIN_POLICY,
                   'policy_path': self.policy_path,
                   'ontology': self.ontology,
                   'goal': args['goal']})

        self.dialogue_counter = 0

    def receive_input(self, inpt):
        """
        Receive input and update the dialogue state.

        :return: Nothing
        """

        # Update dialogue state given the new input
        self.DSTracker.update_state(inpt)

        if self.domain and self.domain in ['CamRest', 'SFH', 'SlotFilling']:
            if self.agent_role == 'system':
                # Perform a database lookup
                db_result, sys_req_slot_entropies = self.db_lookup()

                # Update the dialogue state again to include the database
                # results
                self.DSTracker.update_state_db(
                    db_result=db_result,
                    sys_req_slot_entropies=sys_req_slot_entropies)

            else:
                # Update the dialogue state again to include the system actions
                self.DSTracker.update_state_db(db_result=None, sys_acts=inpt)

        return inpt

    def generate_output(self, args=None):
        """
        Consult the current policy to generate a response.

        :return: List of DialogueAct representing the system's output.
        """
        
        d_state = self.DSTracker.get_state()

        sys_acts = self.policy.next_action(d_state)
        # Copy the sys_acts to be able to iterate over all sys_acts while also
        # replacing some acts
        sys_acts_copy = deepcopy(sys_acts)
        new_sys_acts = []

        # Safeguards to support policies that make decisions on intents only
        # (i.e. do not output slots or values)
        for sys_act in sys_acts:
            if sys_act.intent == 'canthelp' and not sys_act.params:
                slots = \
                    [
                        s for s in d_state.slots_filled if
                        d_state.slots_filled[s]
                    ]
                if slots:
                    slot = random.choice(slots)

                    # Remove the empty canthelp
                    sys_acts_copy.remove(sys_act)

                    new_sys_acts.append(
                        DialogueAct(
                            'canthelp',
                            [DialogueActItem(
                                slot,
                                Operator.EQ,
                                d_state.slots_filled[slot])]))

                else:
                    pass
                    # print('DialogueManager Warning! No slot provided by '
                    #       'policy for canthelp and cannot find a reasonable '
                    #       'one!')
            slots = [x.slot for x in sys_act.params]
            offer_from_inform =  "name" in slots and not self.inform_requested_name
            if sys_act.intent == 'offer' and not sys_act.params:

                build_offer(d_state, new_sys_acts, sys_act, sys_acts,
                            sys_acts_copy)
            elif offer_from_inform:
                build_offer_from_inform(d_state, new_sys_acts, slots)


            elif sys_act.intent == 'inform':
                if self.agent_role == 'system':
                    if sys_act.params and sys_act.params[0].value:
                        continue

                    build_inform(d_state, new_sys_acts, sys_act)

                elif self.agent_role == 'user':
                    if sys_act.params:
                        slot = sys_act.params[0].slot

                        # Do nothing if the slot is already filled
                        if sys_act.params[0].value:
                            continue

                    elif d_state.last_sys_acts and d_state.user_acts and \
                            d_state.user_acts[0].intent == 'request':
                        slot = d_state.user_acts[0].params[0].slot

                    else:
                        slot = \
                            random.choice(
                                list(d_state.user_goal.constraints.keys()))

                    # Populate the inform with a slot from the user goal
                    if d_state.user_goal:
                        # Look for the slot in the user goal
                        if slot in d_state.user_goal.constraints:
                            value = d_state.user_goal.constraints[slot].value
                        else:
                            value = 'dontcare'

                        new_sys_acts.append(
                            DialogueAct(
                                'inform',
                                [DialogueActItem(
                                    slot,
                                    Operator.EQ,
                                    value)]))

                # Remove the empty inform
                sys_acts_copy.remove(sys_act)

            elif sys_act.intent == 'expl-conf':
                if self.agent_role == 'system':
                    if sys_act.params and sys_act.params[0].value:
                        continue

                    build_explicit_confirm(d_state, new_sys_acts, sys_act,
                                                sys_acts_copy)


            elif sys_act.intent == 'request':
                # If the policy did not select a slot
                if not sys_act.params:
                    found = False

                    if self.agent_role == 'system':
                        # Select unfilled slot
                        for slot in d_state.slots_filled:
                            if not d_state.slots_filled[slot]:
                                found = True
                                new_sys_acts.append(
                                    DialogueAct(
                                        'request',
                                        [DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            '')]))
                                break

                    elif self.agent_role == 'user':
                        # Select request from goal
                        if d_state.user_goal:
                            for req in d_state.user_goal.requests:
                                if not d_state.user_goal.requests[req].value:
                                    found = True
                                    new_sys_acts.append(
                                        DialogueAct(
                                            'request',
                                            [DialogueActItem(
                                                req,
                                                Operator.EQ,
                                                '')]))
                                    break

                    if not found:
                        # All slots are filled
                        new_sys_acts.append(
                            DialogueAct(
                                'request',
                                [DialogueActItem(
                                    random.choice(
                                        list(
                                            d_state.slots_filled.keys())[:-1]),
                                    Operator.EQ, '')]))

                    # Remove the empty request
                    sys_acts_copy.remove(sys_act)

        # Append unique new sys acts
        for sa in new_sys_acts:
            if sa not in sys_acts_copy:
                sys_acts_copy.append(sa)

        self.DSTracker.update_state_sysact(sys_acts_copy)

        if len(sys_acts_copy) == 0:
            raise Exception("At least on system act has to be returned!")

        return sys_acts_copy

    def db_lookup(self):
        """
        Perform an SQLite query given the current dialogue state (i.e. given
        which slots have values).

        :return: a dictionary containing the current database results
        """

        # TODO: Add check to assert if each slot in d_state.slots_filled
        # actually exists in the schema.

        d_state = self.DSTracker.get_state()

        # Query the database
        db_result = self.database.db_lookup(d_state)

        if db_result:
            # Calculate entropy of requestable slot values in results -
            # if the flag is off this will be empty
            entropies = \
                dict.fromkeys(self.ontology.ontology['system_requestable'])

            if self.CALCULATE_SLOT_ENTROPIES:
                value_probabilities = {}

                # Count the values
                for req_slot in self.ontology.ontology['system_requestable']:
                    value_probabilities[req_slot] = {}

                    for db_item in db_result:
                        if db_item[req_slot] not in \
                                value_probabilities[req_slot]:
                            value_probabilities[req_slot][
                                db_item[req_slot]] = 1
                        else:
                            value_probabilities[req_slot][
                                db_item[req_slot]] += 1

                # Calculate probabilities
                for slot in value_probabilities:
                    for value in value_probabilities[slot]:
                        value_probabilities[slot][value] /= len(db_result)

                # Calculate entropies
                for slot in entropies:
                    entropies[slot] = 0

                    if slot in value_probabilities:
                        for value in value_probabilities[slot]:
                            entropies[slot] += \
                                value_probabilities[slot][value] * \
                                math.log(value_probabilities[slot][value])

                    entropies[slot] = -entropies[slot]

            return db_result[:self.MAX_DB_RESULTS], entropies

        # Failed to retrieve anything
        # print('Warning! Database call retrieved zero results.')
        return ['empty'], {}

    def restart(self, args):
        """
        Restart the relevant structures or variables, e.g. at the beginning of
        a new dialogue.

        :return: Nothing
        """

        self.DSTracker.initialize(args)
        self.policy.restart(args)
        self.dialogue_counter += 1

    def update_goal(self, goal):
        """
        Update this agent's goal. This is mainly used to propagate the update
        down to the Dialogue State Tracker.

        :param goal: a Goal
        :return: nothing
        """

        if self.DSTracker:
            self.DSTracker.update_goal(goal)
        else:
            print('WARNING: Dialogue Manager goal update failed: No Dialogue '
                  'State Tracker!')

    def get_state(self):
        """
        Get the current dialogue state

        :return: the dialogue state
        """

        return self.DSTracker.get_state()

    def at_terminal_state(self):
        """
        Assess whether the agent is at a terminal state.

        :return: True or False
        """

        return self.DSTracker.get_state().is_terminal()

    def train(self, dialogues):
        """
        Train the policy and dialogue state tracker, if applicable.

        :param dialogues: dialogue experience
        :return: nothing
        """

        if self.TRAIN_POLICY:
            self.policy.train(dialogues)

        if self.TRAIN_DST:
            self.DSTracker.train(dialogues)

    def is_training(self):
        """
        Assess whether there are any trainable components in this Dialogue
        Manager.

        :return: True or False
        """

        return self.TRAIN_DST or self.TRAIN_POLICY

    def load(self, path):
        """
        Load models for the Dialogue State Tracker and Policy.

        :param path: path to the policy model
        :return: nothing
        """

        # TODO: Handle path and loading properly
        self.DSTracker.load('')
        self.policy.load(self.policy_path)

    def save(self):
        """
        Save the models.

        :return: nothing
        """

        if self.DSTracker:
            self.DSTracker.save()

        if self.policy:
            self.policy.save(self.policy_path)
