import random
from copy import deepcopy
from typing import NamedTuple, List

from Dialogue.Action import DialogueAct, DialogueActItem, Operator
from Dialogue.State import SlotFillingDialogueState

STATE_DIM = 45

class Domain(NamedTuple):
    acts_params:List[str]=['inform','request']
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

    return Domain(['inform','request'],dstc2_acts_sys, dstc2_acts_usr,
                         system_requestable_slots, requestable_slots,NActions)

def pick_some(x,num_min,num_max):
    num_to_pick = random.randint(num_min,num_max)
    random.shuffle(x)
    return x[:num_to_pick]

def create_random_dialog_act(domain:Domain,is_system=True):
    acts = []
    if is_system:
        inform_slots = domain.requestable_slots
        request_slots = domain.system_requestable_slots
    else:
        inform_slots = domain.system_requestable_slots
        request_slots = domain.requestable_slots

    intent_p = random.choice(domain.acts_params+[None])
    if intent_p is not None:
        if intent_p == 'inform':
            slots = pick_some(inform_slots,1,3)
        elif intent_p == 'request':
            slots = pick_some(request_slots,1,3)
        else:
            assert False
        act = DialogueAct(intent_p,params=[DialogueActItem(slot,Operator.EQ,None) for slot in slots])
        acts.append(act)

    if is_system:
        intens_w = random.randint(0,len(domain.dstc2_acts_sys))
    else:
        intens_w = random.randint(0,len(domain.dstc2_acts_usr))
    acts.extend([DialogueAct(i) for i in intens_w])