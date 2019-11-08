"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""
from dataclasses import dataclass
from typing import List, Any

__author__ = "Alexandros Papangelis"

from enum import Enum


class Operator(Enum):
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    AND = 7
    OR = 8
    NOT = 9
    IN = 10

    def __str__(self):
        """
        Print the Operator

        :return: a string representation of the Operator
        """
        return f"{self.name}"


class Action:
    def __init__(self):
        self.name = None
        self.funcName = None    # Function name to be called, if applicable?
        self.params = {}        # Dialogue Act Items (slot - operator - value)


"""
Summary Action is a simple class to represent actions in Summary Space. 
"""


class SummaryAction(Enum):
    INFORM_X = 1
    INFORM_XY = 2
    AFFIRM = 3
    AFFIRM_X = 4
    CONFIRM = 5
    CONFIRM_X = 6
    NEGATE = 7
    NEGATE_X = 8
    REQUEST_X = 9
    NOTHING = 10

@dataclass
class DialogueActItem:
    slot:str
    op:Operator
    value:Any

    def __eq__(self, other):
        # TODO: Will need some kind of constraint satisfaction (with tolerance)
        # to efficiently handle all operators
        return self.slot == other.slot and self.op == other.op and \
            self.value == other.value

    def __str__(self):
        """
        Pretty print Dialogue Act Item.

        :return: string
        """

        opr = 'UNK'
        if self.op == Operator.EQ:
            opr = '='
        elif self.op == Operator.NE:
            opr = '!='
        elif self.op == Operator.LT:
            opr = '<'
        elif self.op == Operator.LE:
            opr = '<='
        elif self.op == Operator.GT:
            opr = '>'
        elif self.op == Operator.GE:
            opr = '>='
        elif self.op == Operator.AND:
            opr = 'AND'
        elif self.op == Operator.OR:
            opr = 'OR'
        elif self.op == Operator.NOT:
            opr = 'NOT'
        elif self.op == Operator.IN:
            opr = 'IN'

        result = self.slot

        if self.value:
            result += ' ' + opr + ' ' + self.value

        return result


class DialogueAct(Action):
    """
    Represents a dialogue act, which as a type (e.g. inform, request, etc.)
    and a list of DialogueActItem parameters, which are triplets of
    <slot, operator, value>.
    """
    def __init__(self, intent:str, params:List[DialogueActItem]=[]):
        super(DialogueAct, self).__init__()

        self.name = 'dialogue_act'
        self.intent = intent
        self.params:List[DialogueActItem] = params

    def __eq__(self, other):

        # TODO: Make the check more efficient
        return self.funcName == other.funcName and \
            self.intent == other.intent and \
            self.name == other.name and \
            [s for s in self.params if s not in other.params] == []

    def __str__(self):

        if self.intent:
            return self.intent + \
                   '(' + \
                   ''.join([str(param)+', ' for param in self.params]) + ')'
        else:
            return 'None (DialogueAct)'



"""
The Expression class models complex expressions and defines how to compute 
them.
"""


class Expression:
    
    # An Expression will allow us dialogue acts of the form:
    # inform( 50 < price < 225, food: chinese or italian, ...)
    def __init__(self):
        """
        Not implemented.
        """
        pass



# Represents an event of the (simulated) user tapping onto something in the
# screen.
class TapAct(Action):
    def __init__(self):
        """
        Example, not implemented.
        """
        super(TapAct, self).__init__()
