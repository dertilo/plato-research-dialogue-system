from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from DialogueStateTracker.dummy_dialog_state_tracker import DummyStateTracker
from DialogueManagement.DialoguePolicy.DeepLearning.ReinforcePolicy import (
    ReinforcePolicy,
)

from Domain.Ontology import Ontology
from Domain.DataBase import SQLDataBase

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


def get_num_db_items(database: SQLDataBase):
    cursor = database.SQL_connection.cursor()
    cursor.execute("SELECT * FROM " + database.get_table_name())
    tmp = cursor.fetchall()
    num_db_items = len(tmp)
    return num_db_items


class DialogueManager(ConversationalModule):
    def __init__(
        self,
        settings: dict,
        ontology: Ontology,
        database: SQLDataBase,
        agent_id: int,
        agent_role: str,
        policy_args: dict,
    ):
        """
        Parses the arguments in the dictionary and initializes the appropriate
        models for Dialogue State Tracking and Dialogue Policy.

        :param args: the configuration file parsed into a dictionary
        """

        self.settings = settings
        self.print_level = "debug"
        if "GENERAL" in settings and "print_level" in settings["GENERAL"]:
            self.print_level = settings["GENERAL"]["print_level"]

        self.TRAIN_DST = False
        self.TRAIN_POLICY = False

        self.MAX_DB_RESULTS = 10

        self.DSTracker = None
        self.policy = None
        self.policy_path = None
        self.ontology = None
        assert isinstance(database, SQLDataBase)
        self.database = database
        self.domain = self.settings["DIALOGUE"]["domain"]

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.dialogue_counter = 0
        self.CALCULATE_SLOT_ENTROPIES = True
        self.ontology = ontology

        assert policy_args["type"] == "reinforce"
        alpha, alpha_decay, epsilon, epsilon_decay, gamma = self.get_RL_params(
            policy_args
        )

        self.policy = ReinforcePolicy(
            self.ontology,
            self.database,
            self.agent_id,
            self.agent_role,
            self.domain,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            alpha_decay=alpha_decay,
            epsilon_decay=epsilon_decay,
        )

        if "train" in policy_args:
            self.TRAIN_POLICY = bool(policy_args["train"])

        if "policy_path" in policy_args:
            self.policy_path = policy_args["policy_path"]

        self.DSTracker = DummyStateTracker(self.ontology, self.domain)
        self.load("")

    def get_RL_params(self, policy_args):
        alpha = None
        if "learning_rate" in policy_args:
            alpha = float(policy_args["learning_rate"])
        gamma = None
        if "discount_factor" in policy_args:
            gamma = float(policy_args["discount_factor"])
        epsilon = None
        if "exploration_rate" in policy_args:
            epsilon = float(policy_args["exploration_rate"])
        alpha_decay = None
        if "learning_decay_rate" in policy_args:
            alpha_decay = float(policy_args["learning_decay_rate"])
        epsilon_decay = None
        if "exploration_decay_rate" in policy_args:
            epsilon_decay = float(policy_args["exploration_decay_rate"])
        return alpha, alpha_decay, epsilon, epsilon_decay, gamma

    def initialize(self, args):
        num_db_items = get_num_db_items(self.database)
        self.DSTracker.initialize(num_db_items)
        policy_init_kwargs = {
            "is_training": self.TRAIN_POLICY,
            "policy_path": self.policy_path,
            "ontology": self.ontology,
        }

        if "goal" in args:
            policy_init_kwargs.update({"goal": args["goal"]})
        self.policy.initialize(**policy_init_kwargs)
        self.dialogue_counter = 0

    def receive_input(self, inpt):
        self.DSTracker.update_state(inpt)

        db_result, sys_req_slot_entropies = self.db_lookup()
        self.DSTracker.update_state_db(
            db_result=db_result, sys_req_slot_entropies=sys_req_slot_entropies
        )

        return inpt

    def generate_output(self, args=None):

        d_state = self.DSTracker.get_state()

        sys_acts = self.policy.next_action(d_state)
        # Copy the sys_acts to be able to iterate over all sys_acts while also
        # replacing some acts
        sys_acts_copy = deepcopy(sys_acts)
        new_sys_acts = []

        # Safeguards to support policies that make decisions on intents only
        # (i.e. do not output slots or values)
        for sys_act in sys_acts:
            if not sys_act.params:
                if sys_act.intent == "canthelp":
                    self.cant_help(d_state, new_sys_acts, sys_act, sys_acts_copy)

                elif sys_act.intent == "offer":
                    self.sys_offer(
                        d_state, new_sys_acts, sys_act, sys_acts, sys_acts_copy
                    )

                elif sys_act.intent == "inform" and not sys_act.params[0].value:

                    self.sys_inform(d_state, new_sys_acts, sys_act)
                    sys_acts_copy.remove(sys_act)

                elif sys_act.intent == "request":
                    self.handle_empty_request_action(
                        d_state, new_sys_acts, sys_act, sys_acts_copy
                    )

        # Append unique new sys acts
        for sa in new_sys_acts:
            if sa not in sys_acts_copy:
                sys_acts_copy.append(sa)

        self.DSTracker.update_state_sysact(sys_acts_copy)

        return sys_acts_copy

    def handle_empty_request_action(
        self, d_state, new_sys_acts, sys_act, sys_acts_copy
    ):
        def get_unfilled_slot(d_state):
            for slot in d_state.slots_filled:
                if not d_state.slots_filled[slot]:
                    request_unfilled_slot = DialogueAct(
                        "request", [DialogueActItem(slot, Operator.EQ, "")]
                    )
                    return request_unfilled_slot

            return None

        def build_act_that_requests_random_slot(
            d_state
        ):  # TODO: why would one do such a thing?
            request_random_slot = DialogueAct(
                "request",
                [
                    DialogueActItem(
                        random.choice(list(d_state.slots_filled.keys())[:-1]),
                        Operator.EQ,
                        "",
                    )
                ],
            )
            return request_random_slot

        act = get_unfilled_slot(d_state)
        if act is None:
            act = build_act_that_requests_random_slot(d_state)
        new_sys_acts.append(act)
        # Remove the empty request
        sys_acts_copy.remove(sys_act)

    def sys_inform(self, d_state, new_sys_acts, sys_act):
        if sys_act.params:
            slot = sys_act.params[0].slot
        else:
            slot = d_state.requested_slot
        if not slot:
            slot = random.choice(list(d_state.slots_filled.keys()))
        if d_state.item_in_focus:
            if slot not in d_state.item_in_focus or not d_state.item_in_focus[slot]:
                new_sys_acts.append(
                    DialogueAct(
                        "inform", [DialogueActItem(slot, Operator.EQ, "no info")]
                    )
                )
            else:
                if slot == "name":
                    new_sys_acts.append(
                        DialogueAct(
                            "offer",
                            [
                                DialogueActItem(
                                    slot, Operator.EQ, d_state.item_in_focus[slot]
                                )
                            ],
                        )
                    )
                else:
                    new_sys_acts.append(
                        DialogueAct(
                            "inform",
                            [
                                DialogueActItem(
                                    slot, Operator.EQ, d_state.item_in_focus[slot]
                                )
                            ],
                        )
                    )

        else:
            new_sys_acts.append(
                DialogueAct("inform", [DialogueActItem(slot, Operator.EQ, "no info")])
            )

    def sys_offer(self, d_state, new_sys_acts, sys_act, sys_acts, sys_acts_copy):
        # Remove the empty offer
        sys_acts_copy.remove(sys_act)
        if d_state.item_in_focus:
            new_sys_acts.append(
                DialogueAct(
                    "offer",
                    [
                        DialogueActItem(
                            "name", Operator.EQ, d_state.item_in_focus["name"]
                        )
                    ],
                )
            )

            # Only add these slots if no other acts were output
            # by the DM
            if len(sys_acts) == 1:
                for slot in d_state.slots_filled:
                    if slot in d_state.item_in_focus:
                        if (
                            slot not in ["id", "name"]
                            and slot != d_state.requested_slot
                        ):
                            new_sys_acts.append(
                                DialogueAct(
                                    "inform",
                                    [
                                        DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            d_state.item_in_focus[slot],
                                        )
                                    ],
                                )
                            )
                    else:
                        new_sys_acts.append(
                            DialogueAct(
                                "inform",
                                [DialogueActItem(slot, Operator.EQ, "no info")],
                            )
                        )

    def cant_help(self, d_state, new_sys_acts, sys_act, sys_acts_copy):
        slots = [s for s in d_state.slots_filled if d_state.slots_filled[s]]
        if slots:
            slot = random.choice(slots)

            # Remove the empty canthelp
            sys_acts_copy.remove(sys_act)

            new_sys_acts.append(
                DialogueAct(
                    "canthelp",
                    [DialogueActItem(slot, Operator.EQ, d_state.slots_filled[slot])],
                )
            )

    def db_lookup(self):

        d_state = self.DSTracker.get_state()

        db_result = self.database.db_lookup(d_state)

        if db_result:
            # Calculate entropy of requestable slot values in results -
            # if the flag is off this will be empty
            entropies = self.get_slot_entropies(db_result)

            return db_result[: self.MAX_DB_RESULTS], entropies
        else:
            assert False  # TODO: never happening?

            # Failed to retrieve anything
            # print('Warning! Database call retrieved zero results.')
            return ["empty"], {}

    def get_slot_entropies(self, db_result):
        entropies = dict.fromkeys(self.ontology.ontology["system_requestable"])
        if self.CALCULATE_SLOT_ENTROPIES:
            value_probabilities = {}

            # Count the values
            for req_slot in self.ontology.ontology["system_requestable"]:
                value_probabilities[req_slot] = {}

                for db_item in db_result:
                    if db_item[req_slot] not in value_probabilities[req_slot]:
                        value_probabilities[req_slot][db_item[req_slot]] = 1
                    else:
                        value_probabilities[req_slot][db_item[req_slot]] += 1

            # Calculate probabilities
            for slot in value_probabilities:
                for value in value_probabilities[slot]:
                    value_probabilities[slot][value] /= len(db_result)

            # Calculate entropies
            for slot in entropies:
                entropies[slot] = 0

                if slot in value_probabilities:
                    for value in value_probabilities[slot]:
                        entropies[slot] += value_probabilities[slot][value] * math.log(
                            value_probabilities[slot][value]
                        )

                entropies[slot] = -entropies[slot]
        return entropies

    def restart(self, args):
        num_db_items = get_num_db_items(self.database)
        self.DSTracker.initialize(num_db_items, args)
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
            print(
                "WARNING: Dialogue Manager goal update failed: No Dialogue "
                "State Tracker!"
            )

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
        self.DSTracker.load("")
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
