from abc import ABC, abstractmethod
from copy import deepcopy

from Dialogue.State import SlotFillingDialogueState
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal


def filled_wrongly(gold_slot, state, goal):
    gold_slot_was_filled = gold_slot in state.slots_filled
    with_wrong_value = state.slots_filled[gold_slot] != goal.constraints[gold_slot].value
    gold_value_not_dontcare = goal.constraints[gold_slot].value != 'dontcare'
    return gold_slot_was_filled and \
           with_wrong_value and \
           gold_value_not_dontcare

def request_was_not_done(goal,req):
    return not goal.requests[req].value

class SlotFillingReward(object):
    def __init__(self):
        self.goal = None
        self.turn_penalty = -0.05
        self.failure_penalty = -1
        self.success_reward = 20


    def calculate(self, state, goal=None, force_terminal=False,
                  agent_role='system'):
        """
        Calculate the reward to be assigned for taking action from state.


        :param state: the current state
        :param goal: the agent's goal, used to assess success
        :param force_terminal: force state to be terminal
        :param agent_role: the role of the agent
        :return: a number, representing the calculated reward
        """

        reward = self.turn_penalty
        assert goal is not None
        dialogue_success = False

        if state.is_terminal() or force_terminal:
            # Check that an offer has actually been made
            if state.system_made_offer:
                dialogue_success, reward = self.evaluate_dialog_success(agent_role,
                                                                        goal, reward,
                                                                        state)

            else:
                reward += self.failure_penalty
                dialogue_success = False

        task_success = self.evaluate_task_success(goal, state)

        return reward, dialogue_success, task_success

    def evaluate_task_success(self, goal:Goal, state:SlotFillingDialogueState):
        # Liu & Lane ASRU 2017 Definition of task success
        # We don't care for slots that are not in the goal constraints
        assert goal.ground_truth is None
        if any(filled_wrongly(slot,state,goal) for slot in goal.constraints) \
                or any(request_was_not_done(goal,req) for req in goal.requests):
            task_success = False
        else:
            task_success = True
        return task_success

    def evaluate_dialog_success(self, agent_role, goal, reward, state):
        dialogue_success = True
        # Check that the item offered meets the user's constraints
        for constr in goal.constraints:
            assert goal.ground_truth is None

            if state.item_in_focus:
                # Single-agent case
                if state.item_in_focus[constr] != \
                        goal.constraints[constr].value and \
                        goal.constraints[constr].value != \
                        'dontcare':
                    reward += self.failure_penalty
                    dialogue_success = False
                    break
        # Check that all requests have been addressed
        if dialogue_success:
            not_met = 0

            if agent_role == 'system':
                # Check that the system has responded to all
                # requests (actually) made by the user
                for req in goal.actual_requests:
                    if not goal.actual_requests[req].value:
                        not_met += 1

            elif agent_role == 'user':
                # Check that the user has provided all the
                # requests in the goal
                for req in goal.requests:
                    if not goal.requests[req].value:
                        not_met += 1

            if not_met > 0:
                reward += self.failure_penalty
                dialogue_success = False
            else:
                reward = self.success_reward
        return dialogue_success, reward