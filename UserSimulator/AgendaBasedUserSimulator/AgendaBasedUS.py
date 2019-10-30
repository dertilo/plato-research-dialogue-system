import os
import pickle
import random
from copy import deepcopy

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Domain.DataBase import DataBase, SQLDataBase, JSONDataBase
from Domain.Ontology import Ontology
from UserSimulator.AgendaBasedUserSimulator import Agenda, Goal, ErrorModel
from .. import UserSimulator

"""
AgendaBasedUS is Plato's implementation of the Agenda-Based Simulated Usr, as 
described in: 

Schatzmann, Jost, Blaise Thomson, and Steve Young. 
"Statistical user simulation with a hidden agenda." 
Proc SIGDial, Antwerp 273282.9 (2007).
"""


class AgendaBasedUS(UserSimulator.UserSimulator):
    def __init__(
        self,
        ontology: Ontology,
        database: SQLDataBase,
        user_model,
        patience=3,
        pop_distribution=[1.0],
        slot_confuse_prob=0.0,
        op_confuse_prob=0.0,
        value_confuse_prob=0.0,
        goal_slot_selection_weights=None,
    ):

        super(AgendaBasedUS, self).__init__()

        self.dialogue_turn = 0
        self.policy = None
        self.goals_path = None

        self.user_model = user_model

        self.ontology = ontology
        self.database = database

        self.patience = patience
        self.pop_distribution = pop_distribution
        self.slot_confuse_prob = slot_confuse_prob
        self.op_confuse_prob = op_confuse_prob
        self.value_confuse_prob = value_confuse_prob

        self.goal_slot_selection_weights = goal_slot_selection_weights

        self.curr_patience = self.patience

        self.agenda = Agenda.Agenda()
        self.error_model = ErrorModel.ErrorModel(
            self.ontology,
            self.database,
            self.slot_confuse_prob,
            self.op_confuse_prob,
            self.value_confuse_prob,
        )

        self.goal_generator = Goal.GoalGenerator(
            self.ontology, self.database, self.goals_path
        )
        self.goal: Goal = None
        self.offer_made = False
        self.prev_offer_name = None

        # Store previous system actions to keep track of patience
        self.prev_system_acts = None

    def initialize(self, args):
        """
        Initializes the user simulator, e.g. before each dialogue episode.

        :return: Nothing
        """

        if "goal" not in args:
            # Sample Goal
            goal_slot_selection_weights = None
            if "goal_slot_selection_weights" in args:
                goal_slot_selection_weights = args["goal_slot_selection_weights"]

            self.goal = self.goal_generator.generate(
                goal_slot_selection_weights=goal_slot_selection_weights
            )
        else:
            self.goal = deepcopy(args["goal"])

        # Initialize agenda and user state
        self.agenda.initialize(deepcopy(self.goal))

        self.prev_system_acts = None
        self.curr_patience = self.patience

        self.dialogue_turn = 0
        self.offer_made = False
        self.prev_offer_name = None

    def receive_input(self, inpt, goal=None):
        """
        Receives and processes the input (i.e. the system's response) and
        updates the simulator's internal state.

        :param inpt: list of dialogue acts representing input from the system
        :param goal: the simulated user's goal
        :return: Nothing
        """

        system_acts = inpt

        if goal:
            self.goal = goal

        self.dialogue_turn += 1

        # TODO: AVOID RE-CHECKING THIS IN THE HANDCRAFTED POLICY

        # Update user goal (in ABUS the state is factored into the goal and
        # the agenda)
        for system_act in system_acts:
            if system_act.intent == "offer":
                self.offer_made = True

                # Reset past requests
                if (
                    self.prev_offer_name
                    and system_act.params
                    and system_act.params[0].slot
                    and system_act.params[0].slot == "name"
                    and system_act.params[0].value
                    and self.prev_offer_name != system_act.params[0].value
                ):

                    self.prev_offer_name = system_act.params[0].value

                    self.goal.requests_made = {}

                    for item in self.goal.requests:
                        item.value = ""

        if self.policy:
            self.receive_input_policy(system_acts)
        else:
            self.receive_input_handcrafted(system_acts)

        self.agenda.consistency_check()

    def receive_input_policy(self, system_acts):
        """
        Handle the input according to a policy

        :param system_acts: a list with the system's dialogue acts
        :return: Nothing
        """

        if self.prev_system_acts and self.prev_system_acts == system_acts:
            self.curr_patience -= 1
        else:
            self.curr_patience = self.patience

        self.prev_system_acts = deepcopy(system_acts)

        for system_act in system_acts:
            # 'bye' doesn't seem to appear in the CamRest data
            if (
                system_act.intent == "bye"
                or self.curr_patience == 0
                or self.dialogue_turn > 15
            ):
                self.agenda.push(DialogueAct("bye", []))
                return

            sys_act_slot = (
                "inform" if system_act.intent == "offer" else system_act.intent
            )

            if system_act.params and system_act.params[0].slot:
                sys_act_slot += "_" + system_act.params[0].slot

            # Attempt to recover
            if sys_act_slot not in self.policy:
                if sys_act_slot == "inform_name":
                    sys_act_slot = "offer_name"

            if sys_act_slot not in self.policy:
                if (
                    system_act.intent == "inform"
                    and system_act.params
                    and system_act.params[0].slot in self.goal.constraints
                ):
                    user_act_slots = ["inform_" + system_act.params[0].slot]
                else:
                    print(
                        "Warning! ABUS policy does not know what to do for"
                        " %s" % sys_act_slot
                    )
                    return
            else:
                dacts = list(self.policy[sys_act_slot]["dacts"].keys())
                probs = [self.policy[sys_act_slot]["dacts"][i] for i in dacts]

                user_act_slots = random.choices(dacts, weights=probs)

            for user_act_slot in user_act_slots:
                intent, slot = user_act_slot.split("_")

                if slot == "this" and system_act.params and system_act.params[0].slot:
                    slot = system_act.params[0].slot

                value = ""
                if intent == "inform":
                    if slot in self.goal.constraints:
                        value = self.goal.constraints[slot].value
                    else:
                        value = "dontcare"

                dact = DialogueAct(intent, [DialogueActItem(slot, Operator.EQ, value)])

                self.agenda.remove(dact)
                self.agenda.push(dact)

    def receive_input_handcrafted(self, system_acts):
        """
        Handle the input according to probabilistic rules

        :param system_acts: a list with the system's dialogue acts
        :return: Nothing
        """

        # TODO: Revise these rules wrt other operators (i.e. not only EQ)

        if self.prev_system_acts and self.prev_system_acts == system_acts:
            self.curr_patience -= 1
        else:
            self.curr_patience = self.patience

        self.prev_system_acts = deepcopy(system_acts)

        for system_act in system_acts:
            # Update user goal (in ABUS the state is factored into the goal
            # and the agenda)
            if system_act.intent == "bye" or self.dialogue_turn > 15:
                self.agenda.clear()
                self.agenda.push(DialogueAct("bye", []))

            elif system_act.intent in ["inform", "offer"]:
                # Check that the venue provided meets the constraints
                meets_constraints = True
                for item in system_act.params:
                    if (
                        item.slot in self.goal.constraints
                        and self.goal.constraints[item.slot].value != "dontcare"
                    ):
                        # Remove the inform from the agenda, assuming the
                        # value provided is correct. If it is not, the
                        # act will be pushed again and will be on top of the
                        # agenda (this way we avoid adding / removing
                        # twice.
                        dact = DialogueAct(
                            "inform",
                            [
                                DialogueActItem(
                                    deepcopy(item.slot),
                                    deepcopy(self.goal.constraints[item.slot].op),
                                    deepcopy(self.goal.constraints[item.slot].value),
                                )
                            ],
                        )

                        # Remove and push to make sure the act is on top -
                        # if it already exists
                        self.agenda.remove(dact)

                        if item.value != self.goal.constraints[item.slot].value:
                            meets_constraints = False

                            # For each violated constraint add an inform
                            # TODO: Make this a deny-inform or change
                            # operator to NE

                            self.agenda.push(dact)

                # If it meets the constraints, update the requests
                if meets_constraints:
                    for item in system_act.params:
                        if item.slot in self.goal.requests_made:
                            self.goal.requests_made[item.slot].value = item.value

                            # Mark the value only if the slot has been
                            # requested and is in the requests
                            if item.slot in self.goal.requests:
                                self.goal.requests[item.slot].value = item.value

                            # Remove any requests from the agenda that ask
                            # for that slot
                            # TODO: Revise this for all operators
                            self.agenda.remove(
                                DialogueAct(
                                    "request",
                                    [DialogueActItem(item.slot, Operator.EQ, "")],
                                )
                            )

                # When the system makes a new offer, replace all requests in
                # the agenda
                if system_act.intent == "offer":
                    for r in self.goal.requests:
                        req = deepcopy(self.goal.requests[r])
                        req_dact = DialogueAct("request", [req])

                        # The agenda will replace the old act first
                        self.agenda.push(req_dact)

            # Push appropriate acts into the agenda
            elif system_act.intent == "request":
                if system_act.params:
                    for item in system_act.params:
                        if item.slot in self.goal.constraints:
                            self.agenda.push(
                                DialogueAct(
                                    "inform",
                                    [
                                        DialogueActItem(
                                            deepcopy(item.slot),
                                            deepcopy(
                                                self.goal.constraints[item.slot].op
                                            ),
                                            deepcopy(
                                                self.goal.constraints[item.slot].value
                                            ),
                                        )
                                    ],
                                )
                            )
                        else:
                            self.agenda.push(
                                DialogueAct(
                                    "inform",
                                    [
                                        DialogueActItem(
                                            deepcopy(item.slot), Operator.EQ, "dontcare"
                                        )
                                    ],
                                )
                            )

            # TODO Relax goals if system returns no info for name

    def respond(self):
        """
        Creates the response of the simulated user.

        :return: List of DialogueActs as a response
        """

        if self.curr_patience == 0:
            return [DialogueAct("bye", [])]

        # Sample the number of acts to pop.
        acts = []
        pops = min(
            random.choices(
                range(1, len(self.pop_distribution) + 1), weights=self.pop_distribution
            )[0],
            self.agenda.size(),
        )

        for pop in range(pops):
            act = self.error_model.semantic_noise(self.agenda.pop())

            # Keep track of actual requests made. These are used in reward and
            # success calculation
            if act.intent == "request" and act.params:
                self.goal.requests_made[act.params[0].slot] = act.params[0]

            acts.append(act)

        return acts

    def train(self, data):
        """
        Placeholder for training models

        :param data: dialogue experience
        :return: nothing
        """
        pass

    def save(self, path=None):
        """
        Placeholder for saving trained models

        :param path: path to save the models
        :return: nothing
        """
        pass

    def load(self, path):
        """
        Load an Agenda-Based Simulated Usr policy

        :param path: path to the policy file
        :return: nothing
        """

        if isinstance(path, str):
            if os.path.isfile(path):
                with open(path, "rb") as file:
                    obj = pickle.load(file)

                    if "policy" in obj:
                        self.policy = obj["policy"]

                    print("ABUS policy loaded.")

            else:
                raise FileNotFoundError("ABUS policy file %s not found" % path)
        else:
            raise ValueError("Unacceptable ABUS policy file name: %s " % path)

    def at_terminal_state(self):
        """
        Check if the Agenda-Based Simulated Usr is at a terminal state
        :return: True or False
        """

        return not self.agenda.agenda
