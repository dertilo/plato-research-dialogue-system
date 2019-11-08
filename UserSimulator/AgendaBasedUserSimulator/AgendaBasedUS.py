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

        self.agenda.initialize(deepcopy(self.goal))

        self.prev_system_acts = None
        self.curr_patience = self.patience

        self.dialogue_turn = 0
        self.offer_made = False
        self.prev_offer_name = None

    def receive_input(self, system_acts: List[DialogueAct]):

        self.dialogue_turn += 1

        new_offers = list(filter(self.is_new_offer, system_acts))
        if len(new_offers) > 0:
            self._reset_requests_dueto_new_offers(new_offers)

        self._receive_input_handcrafted(system_acts)

        self.agenda.consistency_check()

    def _reset_requests_dueto_new_offers(self, new_offers: List[DialogueAct]):

        [self._reset_past_requests(offer_act) for offer_act in new_offers]

    def _reset_past_requests(self, system_act):
        self.offer_made = True

        self.prev_offer_name = system_act.params[0].value

        self.goal.requests_made = {}

        for item in self.goal.requests:
            item.value = ""

    def is_new_offer(self, system_act):
        return (
            system_act.intent == "offer"
            and self.prev_offer_name
            and system_act.params
            and system_act.params[0].slot
            and system_act.params[0].slot == "name"
            and system_act.params[0].value
            and self.prev_offer_name != system_act.params[0].value
        )

    def _receive_input_handcrafted(self, system_acts):

        self.alter_patience(system_acts)

        self.prev_system_acts = deepcopy(system_acts)

        for system_act in system_acts:
            # Update user goal (in ABUS the state is factored into the goal
            # and the agenda)
            if system_act.intent == "bye" or self.dialogue_turn > 15:
                self.agenda.clear()
                self.agenda.push(DialogueAct("bye", []))

            elif system_act.intent in ["inform", "offer"]:
                self._handle_inform_and_offer(system_act)

            elif system_act.intent == "request" and len(system_act.params) > 0:
                self._handle_request(system_act.params)

    def alter_patience(self, system_acts):
        system_repeats_itself = (
            self.prev_system_acts and self.prev_system_acts == system_acts
        )
        if system_repeats_itself:
            self.curr_patience -= 1
        else:
            self.curr_patience = self.patience

    def _user_previously_requested_this_slot(self, item):
        return item.slot in self.goal.requests_made

    def _handle_inform_and_offer(self, system_act: DialogueAct):
        # If it meets the constraints, update the requests
        items = system_act.params
        if self.check_that_all_slots_do_meet_constraints_and_update_agenda(items):

            [
                self._update_goal_request_value_remove_from_agenda(item)
                for item in items
                if self._user_previously_requested_this_slot(item)
            ]
        # When the system makes a new offer, replace all requests in
        # the agenda
        elif system_act.intent == "offer":
            self._push_all_goal_request_to_agenda()
        else:
            pass

    def _update_goal_request_value_remove_from_agenda(self, item):
        self.goal.requests_made[item.slot].value = item.value
        # Mark the value only if the slot has been
        # requested and is in the requests
        if item.slot in self.goal.requests:
            self.goal.requests[item.slot].value = item.value
        # Remove any requests from the agenda that ask
        # for that slot
        # TODO: Revise this for all operators
        self.agenda.remove(
            DialogueAct("request", [DialogueActItem(item.slot, Operator.EQ, "")])
        )

    def _user_does_care(self, item, goal_constraints):
        return (
            item.slot in goal_constraints
            and goal_constraints[item.slot].value != "dontcare"
        )

    def check_that_all_slots_do_meet_constraints_and_update_agenda(
        self, items: List[DialogueActItem]
    ):
        # Check that the venue provided meets the constraints
        meets_constraints = all(
            [
                self.remove_from_agenda_push_again_if_false_value(item)
                for item in items
                if self._user_does_care(item, self.goal.constraints)
            ]
        )
        return meets_constraints

    def remove_from_agenda_push_again_if_false_value(self, item):
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
        else:
            meets_constraints = True

        return meets_constraints

    def _push_all_goal_request_to_agenda(self):
        for r in self.goal.requests:
            req = deepcopy(self.goal.requests[r])
            req_dact = DialogueAct("request", [req])

            # The agenda will replace the old act first
            self.agenda.push(req_dact)

    def _handle_request(self, items: List[DialogueActItem]):
        # Push appropriate acts into the agenda
        for item in items:
            system_asks_for_slot_in_goal = item.slot in self.goal.constraints

            if system_asks_for_slot_in_goal:
                operation = deepcopy(self.goal.constraints[item.slot].op)
                slot_value = deepcopy(self.goal.constraints[item.slot].value)
            else:
                operation = Operator.EQ
                slot_value = "dontcare"

            slot_name = deepcopy(item.slot)
            self.agenda.push(
                DialogueAct(
                    "inform", [DialogueActItem(slot_name, operation, slot_value)]
                )
            )

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
