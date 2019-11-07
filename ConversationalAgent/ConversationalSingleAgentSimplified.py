from ConversationalAgent.ConversationalAgent import ConversationalAgent
from UserSimulator.AgendaBasedUserSimulator import Goal

from UserSimulator.AgendaBasedUserSimulator.AgendaBasedUS import AgendaBasedUS
from UserSimulator.AgendaBasedUserSimulator.ErrorModel import ErrorModel
from DialogueManagement import DialogueManager_simplified as DialogueManager
from DialogueManagement.DialoguePolicy.ReinforcementLearning.slot_filling_reward_function import (
    SlotFillingReward,
)
from Utilities.DialogueEpisodeRecorder import DialogueEpisodeRecorder
from Domain import Ontology, DataBase
from Dialogue.Action import DialogueAct

from copy import deepcopy

import os
import random


def validate_configuration(configuration):
    if not configuration["GENERAL"]:
        raise ValueError("Cannot run Plato without GENERAL settings!")

    elif not configuration["GENERAL"]["interaction_mode"]:
        raise ValueError("Cannot run Plato without an " "interaction mode!")

    elif not configuration["DIALOGUE"]:
        raise ValueError("Cannot run Plato without DIALOGUE settings!")

    elif not configuration["AGENT_0"]:
        raise ValueError("Cannot run Plato without at least " "one agent!")


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

        super(ConversationalSingleAgent, self).__init__()

        configuration = configuration

        # There is only one agent in this setting
        self.agent_id = 0

        # Dialogue statistics
        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 50
        self.train_interval = 10
        self.train_epochs = 10

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

        self.agent_goal = None
        self.goal_generator = None

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.recorder = DialogueEpisodeRecorder()

        # TODO: Handle this properly - get reward function type from config
        self.reward_func = SlotFillingReward()

        self.digest_configuration(configuration)

        self.dialogue_manager = DialogueManager.DialogueManager(
            configuration,
            self.ontology,
            self.database,
            self.agent_id,
            "system",
            configuration["AGENT_0"]["DM"]["policy"],
        )

    def digest_configuration(self, configuration):
        validate_configuration(configuration)

        self.build_domain_settings(configuration)
        self.build_general_settings(configuration)
        self.user_simulator = AgendaBasedUS(
            goal_generator=Goal.GoalGenerator(self.ontology, self.database, None),
            error_model=ErrorModel(
                self.ontology,
                slot_confuse_prob=0.0,
                op_confuse_prob=0.0,
                value_confuse_prob=0.0,
            ),
        )

    def build_general_settings(self, configuration):
        if "GENERAL" in configuration and configuration["GENERAL"]:
            if "experience_logs" in configuration["GENERAL"]:
                dialogues_path = None
                if "path" in configuration["GENERAL"]["experience_logs"]:
                    dialogues_path = configuration["GENERAL"]["experience_logs"]["path"]

                if "load" in configuration["GENERAL"]["experience_logs"] and bool(
                    configuration["GENERAL"]["experience_logs"]["load"]
                ):
                    if dialogues_path and os.path.isfile(dialogues_path):
                        self.recorder.load(dialogues_path)
                    else:
                        raise FileNotFoundError(
                            "Dialogue Log file %s not found (did you "
                            "provide one?)" % dialogues_path
                        )

                if "save" in configuration["GENERAL"]["experience_logs"]:
                    self.recorder.set_path(dialogues_path)
                    self.SAVE_LOG = bool(
                        configuration["GENERAL"]["experience_logs"]["save"]
                    )

    def build_domain_settings(self, configuration):
        if "DIALOGUE" in configuration and configuration["DIALOGUE"]:

            if configuration["DIALOGUE"]["domain"]:
                self.domain = configuration["DIALOGUE"]["domain"]

            if configuration["DIALOGUE"]["ontology_path"]:
                if os.path.isfile(configuration["DIALOGUE"]["ontology_path"]):
                    self.ontology = Ontology.Ontology(
                        configuration["DIALOGUE"]["ontology_path"]
                    )
                else:
                    raise FileNotFoundError(
                        "Domain file %s not found"
                        % configuration["DIALOGUE"]["ontology_path"]
                    )

            if configuration["DIALOGUE"]["db_path"]:
                if os.path.isfile(configuration["DIALOGUE"]["db_path"]):
                    if "db_type" in configuration["DIALOGUE"]:
                        if configuration["DIALOGUE"]["db_type"] == "sql":
                            cache_sql_results = False
                            if "cache_sql_results" in configuration["DIALOGUE"]:
                                cache_sql_results = bool(
                                    configuration["DIALOGUE"]["cache_sql_results"]
                                )
                            self.database = DataBase.SQLDataBase(
                                configuration["DIALOGUE"]["db_path"], cache_sql_results
                            )
                        else:
                            self.database = DataBase.DataBase(
                                configuration["DIALOGUE"]["db_path"]
                            )
                    else:
                        # Default to SQL
                        self.database = DataBase.SQLDataBase(
                            configuration["DIALOGUE"]["db_path"]
                        )
                else:
                    raise FileNotFoundError(
                        "Database file %s not found"
                        % configuration["DIALOGUE"]["db_path"]
                    )

            if "goals_path" in configuration["DIALOGUE"]:
                if os.path.isfile(configuration["DIALOGUE"]["goals_path"]):
                    self.goals_path = configuration["DIALOGUE"]["goals_path"]
                else:
                    raise FileNotFoundError(
                        "Goals file %s not found"
                        % configuration["DIALOGUE"]["goals_path"]
                    )

    def initialize(self):

        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0

        self.dialogue_manager.initialize({})

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

    def start_dialogue(self, args=None):

        self.dialogue_turn = 0
        self.user_simulator.initialize()

        self.dialogue_manager.restart({})
        sys_response = [DialogueAct("welcomemsg", [])]

        self.user_simulator.receive_input(sys_response)
        rew, success, task_success = self.reward_func.calculate(
            state=self.dialogue_manager.get_state(), goal=self.user_simulator.goal
        )

        self.recorder.record(
            deepcopy(self.dialogue_manager.get_state()),
            self.dialogue_manager.get_state(),
            sys_response,
            rew,
            success,
            task_success,
        )

        self.dialogue_turn += 1

        self.prev_state = None

        # Re-initialize these for good measure
        self.curr_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.continue_dialogue()

    def continue_dialogue(self):

        usr_input = self.user_simulator.respond()

        self.dialogue_manager.receive_input(usr_input)

        # Keep track of prev_state, for the DialogueEpisodeRecorder
        # Store here because this is the state that the dialogue manager
        # will use to make a decision.
        self.curr_state = deepcopy(self.dialogue_manager.get_state())

        if self.dialogue_turn < self.MAX_TURNS:
            sys_response = self.dialogue_manager.generate_output()

        else:
            sys_response = [DialogueAct("bye", [])]

        self.user_simulator.receive_input(sys_response)
        rew, success, task_success = self.reward_func.calculate(
            state=self.dialogue_manager.get_state(), goal=self.user_simulator.goal
        )

        if self.prev_state:
            self.recorder.record(
                self.prev_state,
                self.curr_state,
                self.prev_action,
                self.prev_reward,
                self.prev_success,
            )

        self.dialogue_turn += 1

        self.prev_state = deepcopy(self.curr_state)
        self.prev_action = deepcopy(sys_response)
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
            task_success=self.prev_task_success,
        )

        if self.dialogue_manager.is_training():
            if (
                self.dialogue_episode % self.train_interval == 0
                and len(self.recorder.dialogues) >= self.minibatch_length
            ):
                for epoch in range(self.train_epochs):

                    # Sample minibatch
                    minibatch = random.sample(
                        self.recorder.dialogues, self.minibatch_length
                    )

                    self.dialogue_manager.train(minibatch)

        self.dialogue_episode += 1
        self.cumulative_rewards += self.recorder.dialogues[-1][-1]["cumulative_reward"]

        if self.dialogue_turn > 0:
            self.total_dialogue_turns += self.dialogue_turn

        if self.dialogue_episode % 10000 == 0:
            self.dialogue_manager.save()

        # Count successful dialogues
        if self.recorder.dialogues[-1][-1]["success"]:
            self.num_successful_dialogues += int(
                self.recorder.dialogues[-1][-1]["success"]
            )

        if self.recorder.dialogues[-1][-1]["task_success"]:
            self.num_task_success += int(
                self.recorder.dialogues[-1][-1]["task_success"]
            )

        # print('OBJECTIVE TASK SUCCESS: {0}'.
        #       format(self.recorder.dialogues[-1][-1]['task_success']))

    def terminated(self):
        """
        Check if this agent is at a terminal state.

        :return: True or False
        """

        return self.dialogue_manager.at_terminal_state()
