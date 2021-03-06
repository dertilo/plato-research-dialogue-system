"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from Domain import Ontology
from DialogueManagement.DialoguePolicy import DialoguePolicy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator

from copy import deepcopy

from typing import List

import random

"""
HandcraftedPolicy is a rule-based system policy, developed as a baseline and as
a quick way to perform sanity checks and debug a Conversational Agent. 
It will try to fill unfilled slots, then suggest an item, and answer any 
requests from the user.
"""


class HandcraftedPolicy(DialoguePolicy.DialoguePolicy):

    def __init__(self, ontology):
        """
        Load the ontology.

        :param ontology: the domain ontology
        """
        super(HandcraftedPolicy, self).__init__()

        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError('Unacceptable ontology type %s ' % ontology)

    def initialize(self, **kwargs):
        """
        Nothing to do here

        :param kwargs:
        :return:
        """
        pass

    def next_action(self, dialogue_state):
        """
        Generate a response given which conditions are met by the current
        dialogue state.

        :param dialogue_state:
        :return:
        """
        # Check for terminal state
        if dialogue_state.is_terminal_state:
            return [DialogueAct('bye', [DialogueActItem('', Operator.EQ, '')])]

        # Check if the user denies and the system has sent an expl-conf in the previous turn
        elif dialogue_state.user_denied_last_sys_acts and \
                dialogue_state.last_sys_acts and \
                'expl-conf' in [x.intent for x in dialogue_state.last_sys_acts]:
            # If the user denies an explicit confirmation, request the slots to be confirmed again
            act: DialogueAct
            request_act = DialogueAct('request', [])
            for act in dialogue_state.last_sys_acts:
                # search the explicit confirmation act in the system acts from the previous turn
                item: DialogueActItem
                if act.intent == 'expl-conf' and act.params:
                    for item in act.params:
                        new_item = DialogueActItem(slot=item.slot, op=Operator.EQ, value=None)
                        request_act.params.append(new_item)

            return [request_act]


        # Check if the user has made any requests
        elif len(dialogue_state.requested_slots) > 0:
            if dialogue_state.item_in_focus and \
                    dialogue_state.system_made_offer:
                requested_slots = dialogue_state.requested_slots

                # Reset request as we attempt to address it
                dialogue_state.requested_slots = []

                items = []
                for rs in requested_slots:
                    value = 'not available'
                    if rs in dialogue_state.item_in_focus and \
                            dialogue_state.item_in_focus[rs]:
                        value = dialogue_state.item_in_focus[rs]

                    items.append(DialogueActItem(rs, Operator.EQ, value))

                return \
                    [DialogueAct(
                        'inform',
                        items)]

            # Else, if no item is in focus or no offer has been made,
            # ignore the user's request

        # Try to fill slots
        requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        if not hasattr(dialogue_state, 'requestable_slot_entropies') or \
                not dialogue_state.requestable_slot_entropies:
            slot = random.choice(requestable_slots)

            while dialogue_state.slots_filled[slot] and \
                    len(requestable_slots) > 1:
                requestable_slots.remove(slot)
                slot = random.choice(requestable_slots)

        else:
            slot = ''
            slots = \
                [k for k, v in
                 dialogue_state.requestable_slot_entropies.items()
                 if v == max(
                    dialogue_state.requestable_slot_entropies.values())
                 and v > 0 and k in requestable_slots]

            if slots:
                slot = random.choice(slots)

                while dialogue_state.slots_filled[slot] \
                        and dialogue_state.requestable_slot_entropies[
                    slot] > 0 \
                        and len(requestable_slots) > 1:
                    requestable_slots.remove(slot)
                    slots = \
                        [k for k, v in
                         dialogue_state.requestable_slot_entropies.items()
                         if v == max(
                            dialogue_state.requestable_slot_entropies.values())
                         and k in requestable_slots]

                    if slots:
                        slot = random.choice(slots)
                    else:
                        break

        if slot and not dialogue_state.slots_filled[slot]:
            return [DialogueAct(
                'request',
                [DialogueActItem(slot, Operator.EQ, '')])]

        elif dialogue_state.item_in_focus:
            name = dialogue_state.item_in_focus['name'] \
                if 'name' in dialogue_state.item_in_focus \
                else 'unknown'

            dacts = [DialogueAct(
                'offer',
                [DialogueActItem('name', Operator.EQ, name)])]

            for slot in dialogue_state.slots_filled:
                if slot != 'requested' and dialogue_state.slots_filled[slot]:
                    if slot in dialogue_state.item_in_focus:
                        if slot not in ['id', 'name']:
                            dacts.append(
                                DialogueAct(
                                    'inform',
                                    [DialogueActItem(
                                        slot,
                                        Operator.EQ,
                                        dialogue_state.item_in_focus[slot])]))
                    else:
                        dacts.append(DialogueAct(
                            'inform',
                            [DialogueActItem(
                                slot,
                                Operator.EQ,
                                'no info')]))

            return dacts

        # if not all filled slot confirmed, try confirmation
        elif not all([v for k, v in dialogue_state.slots_confirmed.items()]):
            # get unconfirmed slots
            unconfirmed_slots = [k for k, v in dialogue_state.slots_confirmed.items() if not v]

            # match match unconfirmed slots with filled slots, as we ask only for confirmation of already filled slots
            confirmation_candidates = []
            for us in unconfirmed_slots:
                if dialogue_state.slots_filled[us]:
                    confirmation_candidates.append(us)

            first_unconfirmed_slot = confirmation_candidates[0]



            return [DialogueAct('expl-conf', [DialogueActItem(first_unconfirmed_slot,
                                                              Operator.EQ,
                                                              dialogue_state.slots_filled[first_unconfirmed_slot])])]

        else:
            # Fallback action - cannot help!
            # Note: We can have this check (no item in focus) at the beginning,
            # but this would assume that the system
            # queried a database before coming in here.
            return [DialogueAct('canthelp', [])]

    def train(self, data):
        """
        Nothing to do here.

        :param data:
        :return:
        """
        pass

    def restart(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

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
