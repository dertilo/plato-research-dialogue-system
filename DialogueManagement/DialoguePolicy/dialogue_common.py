from typing import NamedTuple, List

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState

STATE_DIM = 45

class Domain(NamedTuple):
    dstc2_acts_sys:List[str] = None
    dstc2_acts_usr:List[str] = None
    system_requestable_slots:List[str] = None
    requestable_slots:List[str] = None
    
def encode_state(state:SlotFillingDialogueState,d:Domain):

    def encode_item_in_focus(state):
        # If the agent is a system, then this shows what the top db result is.
        # If the agent is a user, then this shows what information the
        # system has provided
        out = []
        if state.item_in_focus:
            for slot in d.requestable_slots:
                if slot in state.item_in_focus and state.item_in_focus[slot]:
                    out.append(1)
                else:
                    out.append(0)
        else:
            out = [0] * len(d.requestable_slots)
        return out

    def encode_db_matches_ratio(state):
        if state.db_matches_ratio >= 0:
            out = [int(b) for b in
                   format(int(round(state.db_matches_ratio, 2) * 100), '07b')]
        else:
            # If the number is negative (should not happen in general) there
            # will be a minus sign
            out = [int(b) for b in
                   format(int(round(state.db_matches_ratio, 2) * 100),
                          '07b')[1:]]
        assert len(out) == 7
        return out

    def encode_user_acts(state):
        if state.user_acts:
            return [int(b) for b in
                    format(encode_action(state.user_acts, False,d), '05b')]
        else:
            return [0, 0, 0, 0, 0]

    def encode_last_sys_acts(state):
        if state.last_sys_acts:
            integer = encode_action([state.last_sys_acts[0]],True,d)
            # assert integer<16 # TODO(tilo):
            out = [int(b) for b in format(integer, '05b')]
        else:
            out = [0, 0, 0, 0, 0]
        assert len(out) == 5
        return out

    def encode_slots_filled_values(state):
        out = []
        for value in state.slots_filled.values():
            # This contains the requested slot
            out.append(1) if value else out.append(0)
        assert len(out) == 6
        return out

    # --------------------------------------------------------------------------
    temp = []

    temp += [int(b) for b in format(state.turn, '06b')]

    temp += encode_slots_filled_values(state)

    for slot in d.requestable_slots:
        temp.append(1) if slot == state.requested_slot else temp.append(0)

    temp.append(int(state.is_terminal_state))

    temp += encode_item_in_focus(state)
    temp += encode_db_matches_ratio(state)
    temp.append(1) if state.system_made_offer else temp.append(0)
    temp += encode_user_acts(state)
    temp += encode_last_sys_acts(state)

    assert len(temp) == STATE_DIM
    return temp

def encode_action(actions, system,d:Domain                  
                  ):
    """
    Encode the action, given the role. Note that does not have to match
    the agent's role, as the agent may be encoding another agent's action
    (e.g. a system encoding the previous user act).

    :param actions: actions to be encoded
    :param system: whether the role whose action we are encoding is a
                   'system'
    :return: the encoded action
    """

    # TODO: Handle multiple actions
    # TODO: Action encoding in a principled way
    if not actions:
        print('WARNING: Supervised DialoguePolicy action encoding called '
              'with empty actions list (returning -1).')
        return -1

    action = actions[0]
    intent = action.intent

    slot = None
    if action.params and action.params[0].slot:
        slot = action.params[0].slot

    enc = None
    if system:  # encode for system
        if d.dstc2_acts_sys and intent in d.dstc2_acts_sys:
            enc = d.dstc2_acts_sys.index(action.intent)

        elif slot:
            if intent == 'request' and slot in d.system_requestable_slots:
                enc = len(d.dstc2_acts_sys) + d.system_requestable_slots.index(
                    slot)

            elif intent == 'inform' and slot in d.requestable_slots:
                enc = len(d.dstc2_acts_sys) + len(
                    d.system_requestable_slots) + d.requestable_slots.index(slot)
    else:
        if d.dstc2_acts_usr and intent in d.dstc2_acts_usr:
            enc =  d.dstc2_acts_usr.index(action.intent)

        elif slot:
            if intent == 'request' and slot in d.requestable_slots:
                enc = len(d.dstc2_acts_usr) + \
                       d.requestable_slots.index(slot)

            elif action.intent == 'inform' and slot in d.system_requestable_slots:
                enc = len(d.dstc2_acts_usr) + \
                       len(d.requestable_slots) + \
                       d.system_requestable_slots.index(slot)
    if enc is None:
        enc = -1

    return enc

def decode_action(action_enc,d:Domain):
    """
    Decode the action, given the role. Note that does not have to match
    the agent's role, as the agent may be decoding another agent's action
    (e.g. a system decoding the previous user act).

    :param action_enc: action encoding to be decoded
    :param system: whether the role whose action we are decoding is a
                   'system'
    :return: the decoded action
    """
    if action_enc < len(d.dstc2_acts_sys):
        return [DialogueAct(d.dstc2_acts_sys[action_enc], [])]

    if action_enc < len(d.dstc2_acts_sys) + \
            len(d.system_requestable_slots):
        return [DialogueAct(
            'request',
            [DialogueActItem(
                d.system_requestable_slots[
                    action_enc - len(d.dstc2_acts_sys)],
                Operator.EQ,
                '')])]

    if action_enc < len(d.dstc2_acts_sys) + \
            len(d.system_requestable_slots) + \
            len(d.requestable_slots):
        index = \
            action_enc - len(d.dstc2_acts_sys) - \
            len(d.system_requestable_slots)
        return [DialogueAct(
            'inform',
            [DialogueActItem(
                d.requestable_slots[index], Operator.EQ, '')])]
