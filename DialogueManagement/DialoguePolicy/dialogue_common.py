from copy import deepcopy
from typing import NamedTuple, List

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState

STATE_DIM = 45

class Domain(NamedTuple):
    dstc2_acts_sys:List[str] = None
    dstc2_acts_usr:List[str] = None
    system_requestable_slots:List[str] = None
    requestable_slots:List[str] = None
    NActions:int=None
    
def setup_domain(ontology):
    # Extract lists of slots that are frequently used
    informable_slots = \
        deepcopy(list(ontology.ontology['informable'].keys()))
    requestable_slots = \
        deepcopy(ontology.ontology['requestable'])
    system_requestable_slots = \
        deepcopy(ontology.ontology['system_requestable'])

    dstc2_acts_sys = ['offer', 'canthelp', 'affirm',
                           'deny', 'ack', 'bye', 'reqmore',
                           'welcomemsg', 'expl-conf', 'select',
                           'repeat', 'confirm-domain',
                           'confirm']

    # Does not include inform and request that are modelled
    # together with their arguments
    dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack',
                           'thankyou', 'bye', 'reqmore',
                           'hello', 'expl-conf', 'repeat',
                           'reqalts', 'restart', 'confirm']

    dstc2_acts = dstc2_acts_sys
    NActions = len(dstc2_acts)  # system acts without parameters
    NActions += len(
        system_requestable_slots)  # system request with certain slots
    NActions += len(requestable_slots)  # system inform with certain slot

    return Domain(dstc2_acts_sys, dstc2_acts_usr,
                         system_requestable_slots, requestable_slots,NActions)