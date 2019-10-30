"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal

__author__ = "Alexandros Papangelis"

from Dialogue.Action import DialogueAct, DialogueActItem, Operator

"""
The Agenda is a stack-like implementation of the Simulated Usr's agenda. 
It holds DialogueActs and is able to handle complex goals (i.e. goals that 
have sub-goals).
"""


class Agenda:
    def __init__(self):
        self.agenda = []
        self.goal: Goal = None

    def initialize(self, goal: Goal):

        self.goal = goal
        self.clear()

        by_act = DialogueAct("bye", [])
        subgoal_acts = self.handle_subgoals()
        request_acts = [DialogueAct("request", [req]) for req in goal.requests.values()]
        inform_acts = [
            DialogueAct("inform", [constr]) for constr in goal.constraints.values()
        ]

        self.push(by_act)

        for da in subgoal_acts + request_acts + inform_acts:
            self.push(da, force=True)

    def handle_subgoals(self):
        # If there are sub-goals
        # Iterate from last to first because the acts will be popped in
        # reverse order.
        subgoal_acts = []
        for i in range(len(self.goal.subgoals) - 1, -1, -1):
            sg = self.goal.subgoals[i]

            # Acknowledge completion of subgoal
            subgoal_acts.append(DialogueAct("ack_subgoal", []))

            for constr in sg.constraints.values():
                subgoal_acts.append(DialogueAct("inform", [constr]))
        return subgoal_acts

    def push(self, act, force=False):

        if act in self.agenda and not force:
            self.remove(act)

        self.agenda.append(act)

    def pop(self):

        if self.agenda:
            return self.agenda.pop()
        else:
            # TODO: LOG WARNING INSTEAD OF PRINTING
            print("Warning! Attempted to pop an empty agenda.")
            return None

    def peek(self):

        if self.agenda:
            return self.agenda[-1]
        else:
            # TODO: LOG WARNING INSTEAD OF PRINTING
            print("Warning! Attempted to peek an empty agenda.")
            return None

    def remove(self, act):

        if act in self.agenda:
            self.agenda.remove(act)

    def clear(self):

        self.agenda = []

    def consistency_check(self):
        """
        Perform some basic checks to ensure that items in the agenda are
        consistent - i.e. not duplicate, not
        contradicting with current goal, etc.

        :return: Nothing
        """

        # Remove all requests for slots that are filled in the goal
        if self.goal:
            for slot in self.goal.requests_made:
                if self.goal.requests_made[slot].value:
                    self.remove(
                        DialogueAct("request", [DialogueActItem(slot, Operator.EQ, "")])
                    )
        else:
            print(
                "Warning! Agenda consistency check called without goal. "
                "Did you forget to initialize?"
            )

    def size(self):

        return len(self.agenda)
