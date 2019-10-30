import random
from copy import deepcopy
from typing import List

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from UserSimulator.AgendaBasedUserSimulator import Agenda, Goal, ErrorModel

"""
AgendaBasedUS is Plato's implementation of the Agenda-Based Simulated Usr, as 
described in: 

Schatzmann, Jost, Blaise Thomson, and Steve Young. 
"Statistical user simulation with a hidden agenda." 
Proc SIGDial, Antwerp 273282.9 (2007).
"""


class AgendaBasedUS(object):
    def __init__(
        self,
        goal_generator: Goal.GoalGenerator,
        error_model: ErrorModel,
        patience=3,
        pop_distribution=[1.0],
        goal_slot_selection_weights=None,
    ):

        super(AgendaBasedUS, self).__init__()

        self.dialogue_turn = 0
        self.patience = patience
        self.pop_distribution = pop_distribution

        self.goal_slot_selection_weights = goal_slot_selection_weights

        self.curr_patience = self.patience

        self.agenda = Agenda.Agenda()
        self.error_model = error_model

        self.goal_generator = goal_generator
        self.goal: Goal = None
        self.offer_made = False
        self.prev_offer_name = None

        # Store previous system actions to keep track of patience
        self.prev_system_acts = None

    def initialize(self):

        self.goal = self.goal_generator.generate(
            goal_slot_selection_weights=self.goal_slot_selection_weights
        )

        # Initialize agenda and user state
        self.agenda.initialize(deepcopy(self.goal))

        self.prev_system_acts = None
        self.curr_patience = self.patience

        self.dialogue_turn = 0
        self.offer_made = False
        self.prev_offer_name = None

    def receive_input(self, system_acts: List[DialogueAct]):

        self.dialogue_turn += 1
        for system_act in system_acts:
            if system_act.intent == "offer":
                self.handle_offer(system_act)

        self.receive_input_handcrafted(system_acts)

        self.agenda.consistency_check()

    def handle_offer(self, system_act):
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
