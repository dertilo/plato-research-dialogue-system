from Dialogue.State import SlotFillingDialogueState
from Domain import Ontology
from DialogueManagement.DialoguePolicy import DialoguePolicy
from Dialogue.Action import DialogueAct, DialogueActItem, Operator

from copy import deepcopy

import random

"""
HandcraftedPolicy is a rule-based system policy, developed as a baseline and as
a quick way to perform sanity checks and debug a Conversational Agent. 
It will try to fill unfilled slots, then suggest an item, and answer any 
requests from the user.
"""


def get_value(item_in_focus, requested_slot):
    if (
        requested_slot in item_in_focus
        and item_in_focus[requested_slot]
    ):
        value = item_in_focus[requested_slot]
    else:
        value = "not available"
    return value

def build_inform_act(dialogue_state:SlotFillingDialogueState):
    requested_slot = dialogue_state.requested_slot
    # Reset request as we attempt to address it
    dialogue_state.requested_slot = ""
    value = get_value(dialogue_state.item_in_focus, requested_slot)
    dact = [
        DialogueAct("inform", [DialogueActItem(requested_slot, Operator.EQ, value)])
    ]
    return dact


class HandcraftedPolicy(DialoguePolicy.DialoguePolicy):
    def __init__(self, ontology: Ontology.Ontology):
        super(HandcraftedPolicy, self).__init__()
        self.ontology = ontology

    def next_action(self, dialogue_state: SlotFillingDialogueState):
        # Check for terminal state
        if dialogue_state.is_terminal_state:
            dacts = [DialogueAct("bye", [DialogueActItem("", Operator.EQ, "")])]

        # Check if the user has made any requests
        elif (
            dialogue_state.requested_slot
            and dialogue_state.item_in_focus
            and dialogue_state.system_made_offer
        ):
            dacts = build_inform_act(dialogue_state)
        else:
            dacts = self.handle_else(dialogue_state)
        return dacts

    def handle_else(self, dialogue_state):
        # Else, if no item is in focus or no offer has been made,
        # ignore the user's request
        # Try to fill slots
        requestable_slots = deepcopy(self.ontology.ontology["system_requestable"])
        if (
            not hasattr(dialogue_state, "requestable_slot_entropies")
            or not dialogue_state.requestable_slot_entropies
        ):
            slot = random.choice(requestable_slots)

            while dialogue_state.slots_filled[slot] and len(requestable_slots) > 1:
                requestable_slots.remove(slot)
                slot = random.choice(requestable_slots)

        else:
            slot = ""
            slots = [
                k
                for k, v in dialogue_state.requestable_slot_entropies.items()
                if v == max(dialogue_state.requestable_slot_entropies.values())
                and v > 0
                and k in requestable_slots
            ]

            if slots:
                slot = random.choice(slots)

                while (
                    dialogue_state.slots_filled[slot]
                    and dialogue_state.requestable_slot_entropies[slot] > 0
                    and len(requestable_slots) > 1
                ):
                    requestable_slots.remove(slot)
                    slots = [
                        k
                        for k, v in dialogue_state.requestable_slot_entropies.items()
                        if v == max(dialogue_state.requestable_slot_entropies.values())
                        and k in requestable_slots
                    ]

                    if slots:
                        slot = random.choice(slots)
                    else:
                        break
        if slot and not dialogue_state.slots_filled[slot]:
            dacts = [DialogueAct("request", [DialogueActItem(slot, Operator.EQ, "")])]

        elif dialogue_state.item_in_focus:
            name = (
                dialogue_state.item_in_focus["name"]
                if "name" in dialogue_state.item_in_focus
                else "unknown"
            )

            dacts = [DialogueAct("offer", [DialogueActItem("name", Operator.EQ, name)])]

            for slot in dialogue_state.slots_filled:
                if slot != "requested" and dialogue_state.slots_filled[slot]:
                    if slot in dialogue_state.item_in_focus:
                        if slot not in ["id", "name"]:
                            dacts.append(
                                DialogueAct(
                                    "inform",
                                    [
                                        DialogueActItem(
                                            slot,
                                            Operator.EQ,
                                            dialogue_state.item_in_focus[slot],
                                        )
                                    ],
                                )
                            )
                    else:
                        dacts.append(
                            DialogueAct(
                                "inform",
                                [DialogueActItem(slot, Operator.EQ, "no info")],
                            )
                        )
        else:
            # Fallback action - cannot help!
            # Note: We can have this check (no item in focus) at the beginning,
            # but this would assume that the system
            # queried a database before coming in here.
            dacts = [DialogueAct("canthelp", [])]
        return dacts


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
