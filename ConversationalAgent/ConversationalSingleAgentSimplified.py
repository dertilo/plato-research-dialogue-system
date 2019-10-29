
from ConversationalAgent.ConversationalAgent import ConversationalAgent

from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from UserSimulator.DActToLanguageUserSimulator.DTLUserSimulator \
    import DTLUserSimulator
from UserSimulator.UserModel import UserModel
from UserSimulator.AgendaBasedUserSimulator.Goal import GoalGenerator
from DialogueManagement import DialogueManager
from DialogueManagement.DialoguePolicy.ReinforcementLearning.RewardFunction \
    import SlotFillingReward
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder
from Domain import Ontology, DataBase
from NLU.DummyNLU import DummyNLU
from NLU.CamRestNLU import CamRestNLU
from NLG.DummyNLG import DummyNLG
from NLG.CamRestNLG import CamRestNLG
from Dialogue.Action import DialogueAct

from gtts import gTTS
from copy import deepcopy

import os
import random
import speech_recognition as speech_rec


def validate_configuration(configuration):
    if not configuration['GENERAL']:
        raise ValueError('Cannot run Plato without GENERAL settings!')

    elif not configuration['GENERAL']['interaction_mode']:
        raise ValueError('Cannot run Plato without an '
                         'interaction mode!')

    elif not configuration['DIALOGUE']:
        raise ValueError('Cannot run Plato without DIALOGUE settings!')

    elif not configuration['AGENT_0']:
        raise ValueError('Cannot run Plato without at least '
                         'one agent!')

class ConversationalSingleAgent(ConversationalAgent):
    """
    Essentially the dialogue system. Will be able to interact with:

    - Simulated Users via:
        - Dialogue Acts
        - Text

    - Human Users via:
        - Text
        - Speech
        - Online crowd?

    - Data
    """

    def __init__(self, configuration):
        """
        Initialize the internal structures of this agent.

        :param configuration: a dictionary representing the configuration file
        :param agent_id: an integer, this agent's id
        """

        super(ConversationalSingleAgent, self).__init__()

        configuration = configuration
        self.print_level = 'shutup'

        # There is only one agent in this setting
        self.agent_id = 0

        # Dialogue statistics
        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 200
        self.train_interval = 50
        self.train_epochs = 10

        # True values here would imply some default modules
        self.USE_USR_SIMULATOR = False
        self.USER_SIMULATOR_NLU = False
        self.USER_SIMULATOR_NLG = False
        self.USE_NLG = False
        self.USER_HAS_INITIATIVE = True
        self.SAVE_LOG = True

        # The dialogue will terminate after MAX_TURNS (this agent will issue
        # a bye() dialogue act.
        self.MAX_TURNS = 15

        self.dialogue_turn = -1
        self.ontology = None
        self.database = None
        self.domain = None
        self.dialogue_manager = None
        self.user_model = None
        self.user_simulator = None
        self.user_simulator_args = {}
        self.nlu = None
        self.nlg = None

        self.agent_goal = None
        self.goal_generator = None

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.user_model = UserModel()

        self.recorder = DialogueEpisodeRecorder()

        # TODO: Handle this properly - get reward function type from config
        self.reward_func = SlotFillingReward()
        # self.reward_func = SlotFillingGoalAdvancementReward()

        self.digest_configuration(configuration)

        dm_args = dict(
            zip(
                ['settings', 'ontology', 'database', 'domain', 'agent_id',
                 'agent_role'],
                [configuration,
                 self.ontology,
                 self.database,
                 self.domain,
                 self.agent_id,
                 'system'
                 ]
            )
        )
        dm_args.update(configuration['AGENT_0']['DM'])
        self.dialogue_manager = DialogueManager.DialogueManager(dm_args)

    def digest_configuration(self, configuration):
        validate_configuration(configuration)

        self.build_domain_settings(configuration)
        self.build_general_settings(configuration)
        self.build_user_simulator_settings(configuration)

        self.build_NLU_settings(configuration)
        self.build_NLG_settings(configuration)

    def build_NLG_settings(self, configuration):
        if 'NLG' in configuration['AGENT_0'] and \
                configuration['AGENT_0']['NLG'] and \
                configuration['AGENT_0']['NLG']['nlg']:
            if configuration['AGENT_0']['NLG']['nlg'] == 'dummy':
                self.nlg = DummyNLG()

            elif configuration['AGENT_0']['NLG']['nlg'] == 'CamRest':
                if configuration['AGENT_0']['NLG']['model_path']:
                    self.nlg = CamRestNLG(
                        {'model_path':
                             configuration[
                                 'AGENT_0'
                             ]['NLG']['model_path']
                         }
                    )
                else:
                    raise ValueError(
                        'Cannot find model_path in the config.'
                    )

            if self.nlg:
                self.USE_NLG = True

    def build_NLU_settings(self, configuration):
        if 'NLU' in configuration['AGENT_0'] and \
                configuration['AGENT_0']['NLU'] and \
                configuration['AGENT_0']['NLU']['nlu']:
            nlu_args = dict(
                zip(['ontology', 'database'],
                    [self.ontology, self.database]
                    )
            )

            if configuration['AGENT_0']['NLU']['nlu'] == 'dummy':
                self.nlu = DummyNLU(nlu_args)

            elif configuration['AGENT_0']['NLU']['nlu'] == 'CamRest':
                if configuration['AGENT_0']['NLU']['model_path']:
                    nlu_args['model_path'] = \
                        configuration['AGENT_0']['NLU']['model_path']
                    self.nlu = CamRestNLU(nlu_args)
                else:
                    raise ValueError(
                        'Cannot find model_path in the config.'
                    )

    def build_user_simulator_settings(self, configuration):
        if 'USER_SIMULATOR' in configuration['AGENT_0']:
            # Agent 0 simulator configuration
            a0_sim_config = configuration['AGENT_0']['USER_SIMULATOR']
            if a0_sim_config and a0_sim_config['simulator']:
                # Default settings
                self.user_simulator_args['ontology'] = self.ontology
                self.user_simulator_args['database'] = self.database
                self.user_simulator_args['um'] = self.user_model
                self.user_simulator_args['patience'] = 5

                if a0_sim_config['simulator'] == 'agenda':
                    if 'patience' in a0_sim_config:
                        self.user_simulator_args['patience'] = \
                            int(a0_sim_config['patience'])

                    if 'pop_distribution' in a0_sim_config:
                        if isinstance(
                                a0_sim_config['pop_distribution'], list
                        ):
                            self.user_simulator_args['pop_distribution'] = \
                                a0_sim_config['pop_distribution']
                        else:
                            self.user_simulator_args['pop_distribution'] = \
                                eval(a0_sim_config['pop_distribution'])

                    if 'slot_confuse_prob' in a0_sim_config:
                        self.user_simulator_args['slot_confuse_prob'] = \
                            float(a0_sim_config['slot_confuse_prob'])
                    if 'op_confuse_prob' in a0_sim_config:
                        self.user_simulator_args['op_confuse_prob'] = \
                            float(a0_sim_config['op_confuse_prob'])
                    if 'value_confuse_prob' in a0_sim_config:
                        self.user_simulator_args['value_confuse_prob'] = \
                            float(a0_sim_config['value_confuse_prob'])

                    if 'goal_slot_selection_weights' in a0_sim_config:
                        self.user_simulator_args[
                            'goal_slot_selection_weights'
                        ] = a0_sim_config['goal_slot_selection_weights']

                    if 'nlu' in a0_sim_config:
                        self.user_simulator_args['nlu'] = \
                            a0_sim_config['nlu']

                        if self.user_simulator_args['nlu'] == 'dummy':
                            self.user_simulator_args['database'] = \
                                self.database

                        self.USER_SIMULATOR_NLU = True

                    if 'nlg' in a0_sim_config:
                        self.user_simulator_args['nlg'] = \
                            a0_sim_config['nlg']

                        if self.user_simulator_args['nlg'] == 'CamRest':
                            if a0_sim_config:
                                self.user_simulator_args[
                                    'nlg_model_path'
                                ] = a0_sim_config['nlg_model_path']

                                self.USER_SIMULATOR_NLG = True

                            else:
                                raise ValueError(
                                    'Usr Simulator NLG: Cannot find '
                                    'model_path in the config.'
                                )

                        elif self.user_simulator_args['nlg'] == 'dummy':
                            self.USER_SIMULATOR_NLG = True

                    if 'goals_file' in a0_sim_config:
                        self.user_simulator_args['goals_file'] = \
                            a0_sim_config['goals_file']

                    if 'policy_file' in a0_sim_config:
                        self.user_simulator_args['policy_file'] = \
                            a0_sim_config['policy_file']

                    self.user_simulator = AgendaBasedUS(
                        self.user_simulator_args
                    )

                elif a0_sim_config['simulator'] == 'dtl':
                    if 'policy_file' in a0_sim_config:
                        self.user_simulator_args['policy_file'] = \
                            a0_sim_config['policy_file']
                        self.user_simulator = DTLUserSimulator(
                            self.user_simulator_args
                        )
                    else:
                        raise ValueError(
                            'Error! Cannot start DAct-to-Language '
                            'simulator without a policy file!'
                        )

            else:
                # Fallback to agenda based simulator with default settings
                self.user_simulator = AgendaBasedUS(
                    self.user_simulator_args
                )

    def build_general_settings(self, configuration):
        if 'GENERAL' in configuration and \
                configuration['GENERAL']:
            if 'experience_logs' in configuration['GENERAL']:
                dialogues_path = None
                if 'path' in \
                        configuration['GENERAL']['experience_logs']:
                    dialogues_path = \
                        configuration['GENERAL'][
                            'experience_logs']['path']

                if 'load' in \
                        configuration['GENERAL']['experience_logs'] \
                        and bool(
                    configuration['GENERAL'][
                        'experience_logs']['load']
                ):
                    if dialogues_path and os.path.isfile(dialogues_path):
                        self.recorder.load(dialogues_path)
                    else:
                        raise FileNotFoundError(
                            'Dialogue Log file %s not found (did you '
                            'provide one?)' % dialogues_path)

                if 'save' in \
                        configuration['GENERAL']['experience_logs']:
                    self.recorder.set_path(dialogues_path)
                    self.SAVE_LOG = bool(
                        configuration['GENERAL'][
                            'experience_logs']['save']
                    )

            if configuration['GENERAL']['interaction_mode'] == \
                    'simulation':
                self.USE_USR_SIMULATOR = True

    def build_domain_settings(self,configuration):
        if 'DIALOGUE' in configuration and \
                configuration['DIALOGUE']:
            if 'initiative' in configuration['DIALOGUE']:
                self.USER_HAS_INITIATIVE = bool(
                    configuration['DIALOGUE']['initiative'] == 'user'
                )
                self.user_simulator_args['us_has_initiative'] = \
                    self.USER_HAS_INITIATIVE

            if configuration['DIALOGUE']['domain']:
                self.domain = configuration['DIALOGUE']['domain']

            if configuration['DIALOGUE']['ontology_path']:
                if os.path.isfile(
                        configuration['DIALOGUE']['ontology_path']
                ):
                    self.ontology = Ontology.Ontology(
                        configuration['DIALOGUE']['ontology_path']
                    )
                else:
                    raise FileNotFoundError(
                        'Domain file %s not found' %
                        configuration['DIALOGUE']['ontology_path'])

            if configuration['DIALOGUE']['db_path']:
                if os.path.isfile(
                        configuration['DIALOGUE']['db_path']
                ):
                    if 'db_type' in configuration['DIALOGUE']:
                        if configuration['DIALOGUE']['db_type'] == \
                                'sql':
                            cache_sql_results = False
                            if 'cache_sql_results' in configuration[
                                'DIALOGUE']:
                                cache_sql_results = bool(
                                    configuration['DIALOGUE'][
                                        'cache_sql_results'])
                            self.database = DataBase.SQLDataBase(
                                configuration['DIALOGUE']['db_path'],
                                cache_sql_results
                            )
                        else:
                            self.database = DataBase.DataBase(
                                configuration['DIALOGUE']['db_path']
                            )
                    else:
                        # Default to SQL
                        self.database = DataBase.SQLDataBase(
                            configuration['DIALOGUE']['db_path']
                        )
                else:
                    raise FileNotFoundError(
                        'Database file %s not found' %
                        configuration['DIALOGUE']['db_path']
                    )

            if 'goals_path' in configuration['DIALOGUE']:
                if os.path.isfile(
                        configuration['DIALOGUE']['goals_path']
                ):
                    self.goals_path = \
                        configuration['DIALOGUE']['goals_path']
                else:
                    raise FileNotFoundError(
                        'Goals file %s not found' %
                        configuration['DIALOGUE']['goals_path']
                    )

    def initialize(self):

        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0

        if self.nlu:
            self.nlu.initialize({})

        self.dialogue_manager.initialize({})

        if self.nlg:
            self.nlg.initialize({})

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

    def start_dialogue(self, args=None):
        """
        Perform initial dialogue turn.

        :param args: optional args
        :return:
        """

        self.dialogue_turn = 0
        sys_utterance = ''

        if self.USE_USR_SIMULATOR:
            self.user_simulator.initialize(self.user_simulator_args)

            if self.print_level in ['debug']:
                print('DEBUG > Usr goal:')
                print(self.user_simulator.goal)

        self.dialogue_manager.restart({})

        if not self.USER_HAS_INITIATIVE:
            # sys_response = self.dialogue_manager.respond()
            sys_response = [DialogueAct('welcomemsg', [])]

            if self.USE_NLG:
                sys_utterance = self.nlg.generate_output(
                    {'dacts': sys_response}
                )
                if self.print_level in ['debug']:
                    print('SYSTEM > %s ' % sys_utterance)

            else:
                if self.print_level in ['debug']:
                    print(
                        'SYSTEM > %s ' % '; '.
                        join([str(sr) for sr in sys_response])
                    )

            if self.USE_USR_SIMULATOR:
                usim_input = sys_response

                if self.USER_SIMULATOR_NLU and self.USE_NLG:
                    usim_input = self.user_simulator.nlu.process_input(
                        sys_utterance
                    )

                self.user_simulator.receive_input(usim_input)
                rew, success, task_success = self.reward_func.calculate(
                    self.dialogue_manager.get_state(),
                    sys_response,
                    self.user_simulator.goal
                )
            else:
                rew, success, task_success = 0, None, None

            self.recorder.record(
                deepcopy(self.dialogue_manager.get_state()),
                self.dialogue_manager.get_state(),
                sys_response,
                rew,
                success,
                task_success,
                output_utterance=sys_utterance
            )

            self.dialogue_turn += 1

        self.prev_state = None

        # Re-initialize these for good measure
        self.curr_state = None
        self.prev_usr_utterance = None
        self.prev_sys_utterance = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.continue_dialogue()

    def continue_dialogue(self):
        """
        Perform next dialogue turn.

        :return: nothing
        """

        usr_utterance = ''
        sys_utterance = ''

        if self.USE_USR_SIMULATOR:
            usr_input = self.user_simulator.respond()

            # TODO: THIS FIRST IF WILL BE HANDLED BY ConversationalAgentGeneric
            #  -- SHOULD NOT LIVE HERE
            if isinstance(self.user_simulator, DTLUserSimulator):
                if self.print_level in ['debug']:
                    print('USER (NLG) > %s \n' % usr_input)
                usr_input = self.nlu.process_input(
                    usr_input,
                    self.dialogue_manager.get_state()
                )

            elif self.USER_SIMULATOR_NLG:
                if self.print_level in ['debug']:
                    print('USER > %s \n' % usr_input)

                if self.nlu:
                    usr_input = self.nlu.process_input(usr_input)

                    # Otherwise it will just print the user's NLG but use the
                    # simulator's output DActs to proceed.

            else:
                if self.print_level in ['debug']:
                    print('USER (DACT) > %s \n' % usr_input[0])

        else:
            usr_utterance = input('USER > ')

            # Process the user's utterance
            if self.nlu:
                usr_input = self.nlu.process_input(
                    usr_utterance,
                    self.dialogue_manager.get_state()
                )
            else:
                raise EnvironmentError(
                    'ConversationalAgent: No NLU defined for '
                    'text-based interaction!'
                )

        self.dialogue_manager.receive_input(usr_input)

        # Keep track of prev_state, for the DialogueEpisodeRecorder
        # Store here because this is the state that the dialogue manager
        # will use to make a decision.
        self.curr_state = deepcopy(self.dialogue_manager.get_state())


        if self.dialogue_turn < self.MAX_TURNS:
            sys_response = self.dialogue_manager.generate_output()

        else:
            sys_response = [DialogueAct('bye', [])]

        if self.USE_NLG:
            sys_utterance = self.nlg.generate_output({'dacts': sys_response})
            if self.print_level in ['debug']:
                print('SYSTEM > %s ' % sys_utterance)

        else:
            if self.print_level in ['debug']:
                print('SYSTEM > %s ' % '; '.join([str(sr) for sr in sys_response]))

        if self.USE_USR_SIMULATOR:
            usim_input = sys_response

            if self.USER_SIMULATOR_NLU and self.USE_NLG:
                usim_input = \
                    self.user_simulator.nlu.process_input(sys_utterance)
                if self.print_level in ['debug']:
                    print(
                        'USER NLU '
                        '> %s ' % '; '.join([str(ui) for ui in usim_input])
                    )

            self.user_simulator.receive_input(usim_input)
            rew, success, task_success = \
                self.reward_func.calculate(
                    self.dialogue_manager.get_state(),
                    sys_response,
                    self.user_simulator.goal
                )
        else:
            rew, success, task_success = 0, None, None

        if self.prev_state:
            self.recorder.record(
                self.prev_state,
                self.curr_state,
                self.prev_action,
                self.prev_reward,
                self.prev_success,
                input_utterance=usr_utterance,
                output_utterance=sys_utterance
            )

        self.dialogue_turn += 1

        self.prev_state = deepcopy(self.curr_state)
        self.prev_action = deepcopy(sys_response)
        self.prev_usr_utterance = deepcopy(usr_utterance)
        self.prev_sys_utterance = deepcopy(sys_utterance)
        self.prev_reward = rew
        self.prev_success = success
        self.prev_task_success = task_success

    def end_dialogue(self):
        """
        Perform final dialogue turn. Train and save models if applicable.

        :return: nothing
        """

        # Record final state
        self.recorder.record(
            self.curr_state,
            self.curr_state,
            self.prev_action,
            self.prev_reward,
            self.prev_success,
            input_utterance=self.prev_usr_utterance,
            output_utterance=self.prev_sys_utterance,
            task_success=self.prev_task_success
        )

        if self.dialogue_manager.is_training():
            if self.dialogue_episode % self.train_interval == 0 and \
                    len(self.recorder.dialogues) >= self.minibatch_length:
                for epoch in range(self.train_epochs):
                    if self.print_level in ['debug']:
                        print('Training epoch {0} of {1}'.format(
                            epoch,
                            self.train_epochs)
                        )

                    # Sample minibatch
                    minibatch = random.sample(
                        self.recorder.dialogues,
                        self.minibatch_length
                    )

                    if self.nlu:
                        self.nlu.train(minibatch)

                    self.dialogue_manager.train(minibatch)

                    if self.nlg:
                        self.nlg.train(minibatch)

        self.dialogue_episode += 1
        self.cumulative_rewards += \
            self.recorder.dialogues[-1][-1]['cumulative_reward']
        if self.print_level in ['debug']:
            print('CUMULATIVE REWARD: {0}'.format(self.recorder.dialogues[-1][-1]['cumulative_reward']))

        if self.dialogue_turn > 0:
            self.total_dialogue_turns += self.dialogue_turn

        if self.dialogue_episode % 10000 == 0:
            self.dialogue_manager.save()

        # Count successful dialogues
        if self.recorder.dialogues[-1][-1]['success']:
            if self.print_level in ['debug']:
                print('SUCCESS (Subjective)!')
            self.num_successful_dialogues += \
                int(self.recorder.dialogues[-1][-1]['success'])

        else:
            if self.print_level in ['debug']:
                print('FAILURE (Subjective).')

        if self.recorder.dialogues[-1][-1]['task_success']:
            self.num_task_success += \
                int(self.recorder.dialogues[-1][-1]['task_success'])

        # print('OBJECTIVE TASK SUCCESS: {0}'.
        #       format(self.recorder.dialogues[-1][-1]['task_success']))

    def terminated(self):
        """
        Check if this agent is at a terminal state.

        :return: True or False
        """

        return self.dialogue_manager.at_terminal_state()

