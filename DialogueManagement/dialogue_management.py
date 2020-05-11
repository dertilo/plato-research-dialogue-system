import random
from Dialogue.Action import DialogueAct, DialogueActItem, Operator


def build_offer_from_inform(d_state, new_sys_acts, slots):

    new_sys_acts.append(
        DialogueAct(
            "offer",
            [DialogueActItem("name", Operator.EQ, d_state.item_in_focus["name"])],
        )
    )

    new_sys_acts.append(
        DialogueAct(
            "inform",
            [
                DialogueActItem(slot, Operator.EQ, d_state.item_in_focus[slot])
                for slot in slots
                if slot is not "name"
            ],
        )
    )


def build_offer(d_state, new_sys_acts, sys_act, sys_acts, sys_acts_copy):
    # TODO: What should happen if a policy selects offer, but there is no item in focus?
    if d_state.item_in_focus:
        # Remove the empty offer
        sys_acts_copy.remove(sys_act)

        new_sys_acts.append(
            DialogueAct(
                "offer",
                [DialogueActItem("name", Operator.EQ, d_state.item_in_focus["name"])],
            )
        )
        assert len(sys_acts)==1
        # Only add these slots if no other acts were output
        # by the DM
        if len(sys_acts) == 1:
            for slot in d_state.slots_filled:
                if slot in d_state.item_in_focus:
                    if (
                        slot not in ["id", "name"]
                        and slot not in d_state.requested_slots
                    ):
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
                        DialogueAct(
                            "inform", [DialogueActItem(slot, Operator.EQ, "no info")]
                        )
                    )


def build_inform(d_state, new_sys_acts, sys_act):
    slots = []
    if sys_act.params:
        # use the slots addressed by the inform act (slots selected by the policy)
        slots = [x.slot for x in sys_act.params]
    else:
        # use the requested slots, if any available
        if d_state.requested_slots:
            slots = [x for x in d_state.requested_slots]
    if not slots:
        # if we still have no slot(s) use one from the filled slots
        slots = [random.choice(list(d_state.slots_filled.keys()))]

    def get_value(slot):
        if not d_state.item_in_focus or (
            slot not in d_state.item_in_focus or not d_state.item_in_focus[slot]
        ):
            value = "no info"
        else:
            value = d_state.item_in_focus[slot]
        return value

    act_items = [DialogueActItem(slot, Operator.EQ, get_value(slot)) for slot in slots]
    new_sys_acts.append(DialogueAct("inform", act_items))


def build_explicit_confirm(d_state, new_sys_acts, sys_act, sys_acts_copy):
    slots = []
    if sys_act.params:
        # use the slots addressed by the expl-conf act (slots selected by the policy)
        slots = [x.slot for x in sys_act.params]
    if len(slots) == 0:
        # if no slot was selected by the policy, do randomly select one
        random_slot = random.choices(list(d_state.slots_filled.keys()))
        slots.extend(random_slot)

    def _build_expl_confirm_act_item(slot):
        value = d_state.slots_filled[slot] if d_state.slots_filled[slot] else "no info"
        item = DialogueActItem(slot, Operator.EQ, value)
        return item

    new_sys_acts.append(
        DialogueAct("expl-conf", [_build_expl_confirm_act_item(slot) for slot in slots])
    )
    # Remove the empty expl-conf
    sys_acts_copy.remove(sys_act)
