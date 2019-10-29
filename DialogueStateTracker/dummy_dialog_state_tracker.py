
from Dialogue.State import SlotFillingDialogueState
from abc import ABC, abstractmethod

from Domain.Ontology import Ontology
from copy import deepcopy



class DummyStateTracker(object):

    def __init__(self, ontology,domain):

        super(DummyStateTracker, self).__init__()

        self.ontology = None
        if isinstance(ontology, Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = Ontology(ontology)
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

        self.domain = domain
        if domain in ['CamRest', 'SlotFilling']:
            self.DState = \
                SlotFillingDialogueState(
                    {'slots': self.ontology.ontology['system_requestable']})
        else:
            print('Warning! Domain has not been defined. Using Slot-Filling '
                  'Dialogue State')
            self.DState = \
                SlotFillingDialogueState(
                    {'slots': self.ontology.ontology['system_requestable']})

    def initialize(self,num_db_items, args=None):

        self.DB_ITEMS = num_db_items

        if self.DB_ITEMS <= 0:
            print('Warning! DST could not get number of DB items.')
            self.DB_ITEMS = 110     # Default for CamRestaurants

        self.DState.initialize(args)
        # No constraints have been expressed yet
        self.DState.db_matches_ratio = 1.0

        self.DState.turn = 0

    def update_state(self, dacts):
        """
        Update the dialogue state given the input dialogue acts. This function
        basically tracks which intents, slots, and values have been mentioned
        and updates the dialogue state accordingly.

        :param dacts: a list of dialogue acts (usually the output of NLU)
        :return: the updated dialogue state
        """

        # TODO: These rules will create a field in the dialogue state slots
        # filled dictionary if one doesn't exist.
        self.DState.user_acts = deepcopy(dacts)

        # Reset past request
        self.DState.requested_slot = ''

        for dact in dacts:
            if dact.intent in ['inform', 'offer']:
                # The user provided new information so the system hasn't made
                # any offers taking that into account yet.
                # self.DState.system_made_offer = False

                if dact.intent == 'offer':
                    self.DState.system_made_offer = True

                for dact_item in dact.params:
                    if dact_item.slot in self.DState.slots_filled:
                        self.DState.slots_filled[dact_item.slot] = \
                            dact_item.value

                    elif self.DState.user_goal:
                        if dact_item.slot in \
                                self.DState.user_goal.actual_requests:
                            self.DState.user_goal.actual_requests[
                                dact_item.slot].value = dact_item.value

                        # Only update requests that have been asked for
                        if dact_item.slot in self.DState.user_goal.requests:
                            self.DState.user_goal.requests[
                                dact_item.slot].value = dact_item.value

            elif dact.intent == 'request':
                for dact_item in dact.params:
                    # TODO: THIS WILL ONLY SAVE THE LAST DACT ITEM! --
                    # THIS APPLIES TO THE FOLLOWING RULES AS WELL

                    if dact_item.slot == 'slot' and dact_item.value:
                        # Case where we have request(slot = slot_name)
                        self.DState.requested_slot = dact_item.value
                    else:
                        # Case where we have: request(slot_name)
                        self.DState.requested_slot = dact_item.slot

            elif dact.intent == 'bye':
                self.DState.is_terminal_state = True

        # Increment turn
        self.DState.turn += 1

        return self.DState

    def update_state_db(self, db_result=None, sys_req_slot_entropies=None,
                        sys_acts=None):
        """
        This is a special function that is mostly designed for the multi-agent
        setup. If the state belongs to a 'system' agent, then this function
        will update the current database results. If the state belongs to a
        'user' agent, then this function will update the 'item in focus' fields
        of the dialogue state, given the last system action.

        :param db_result: a dictionary containing the database query results
        :param sys_req_slot_entropies: calculated entropies for requestable
                                       slots
        :param sys_acts: the system's acts
        :return:
        """

        if db_result and sys_acts:
            raise ValueError('Dialogue State Tracker: Cannot update state as '
                             'both system and user (i.e. please use only one '
                             'argument as appropriate).')

        # This should be called if the agent is a system
        if db_result:
            self.DState.db_matches_ratio = \
                float(len(db_result) / self.DB_ITEMS)

            if db_result[0] == 'empty':
                self.DState.item_in_focus = []

            else:
                self.DState.item_in_focus = db_result[0]

            if sys_req_slot_entropies:
                self.DState.system_requestable_slot_entropies = \
                    deepcopy(sys_req_slot_entropies)

            self.DState.db_result = db_result

        # This should be called if the agent is a user
        elif sys_acts:
            # Create dictionary if it doesn't exist or reset it if a new offer
            # has been made
            if not self.DState.item_in_focus or \
                    'offer' in [a.intent for a in sys_acts]:
                self.DState.item_in_focus = \
                    dict.fromkeys(self.ontology.ontology['requestable'])

            for sys_act in sys_acts:
                if sys_act.intent in ['inform', 'offer']:
                    for item in sys_act.params:
                        self.DState.item_in_focus[item.slot] = item.value

                        if self.DState.user_goal:
                            if item.slot in \
                                    self.DState.user_goal.actual_requests:
                                self.DState.user_goal.actual_requests[
                                    item.slot].value = item.value

                            # Only update requests that have been asked for
                            if item.slot in self.DState.user_goal.requests:
                                self.DState.user_goal.requests[
                                    item.slot].value = item.value

        return self.DState

    def update_state_sysact(self, sys_acts):
        """
        Updates the last system act and the goal, given that act. This is
        useful as we may want to update parts of the state given NLU output
        and then update again once the system produces a response.

        :param sys_acts: the last system acts
        :return:
        """

        if sys_acts:
            self.DState.last_sys_acts = sys_acts

            for sys_act in sys_acts:
                if sys_act.intent == 'offer':
                    self.DState.system_made_offer = True

                # Keep track of actual requests made. These are used in reward
                # and success calculation for systems. The
                # reasoning is that it does not make sense to penalise a system
                # for an unanswered request that was
                # never actually made by the user.
                # If the current agent is a system then these will be
                # disregarded.
                if sys_act.intent == 'request' and sys_act.params and \
                        self.DState.user_goal:
                    self.DState.user_goal.actual_requests[
                        sys_act.params[0].slot] = sys_act.params[0]

                # Similarly, keep track of actual constraints made.
                if sys_act.intent == 'inform' and sys_act.params and \
                        self.DState.user_goal:
                    self.DState.user_goal.actual_constraints[
                        sys_act.params[0].slot] = sys_act.params[0]

                # Reset the request if the system asks for more information,
                # assuming that any previously offered item
                # is now invalid.
                # elif sys_act.intent == 'request':
                #     self.DState.system_made_offer = False

    def update_goal(self, goal):
        """
        Updates the agent's goal

        :param goal: a Goal
        :return:
        """

        # TODO: Do a deep copy?
        self.DState.user_goal = goal

    def train(self, data):
        """
        Nothing to do here.

        :param data:
        :return:
        """
        pass
    
    def get_state(self):
        """
        Returns the current dialogue state.

        :return: the current dialogue state
        """
        return self.DState

    def save(self, path=None):
        """
        Nothing to do here.

        :param path:
        :return:
        """
        pass

    def load(self, path):
        """
        Nothing to do here.

        :param path:
        :return:
        """
        pass
