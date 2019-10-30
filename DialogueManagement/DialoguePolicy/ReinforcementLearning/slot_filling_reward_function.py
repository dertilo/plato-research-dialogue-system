from abc import ABC, abstractmethod
from copy import deepcopy

from Dialogue.State import SlotFillingDialogueState
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal


def filled_wrongly(gold_slot, state, goal):
    gold_slot_was_filled = gold_slot in state.slots_filled
    with_wrong_value = (
        state.slots_filled[gold_slot] != goal.constraints[gold_slot].value
    )
    gold_value_not_dontcare = goal.constraints[gold_slot].value != "dontcare"
    return gold_slot_was_filled and with_wrong_value and gold_value_not_dontcare


def request_was_not_done(goal, req):
    return not goal.requests[req].value


class SlotFillingReward(object):
    def __init__(self):
        self.goal = None
        self.turn_penalty = -0.05
        self.failure_penalty = -1
        self.success_reward = 20

    def calculate(self, state: SlotFillingDialogueState, goal: Goal):

        assert goal is not None
        dialogue_success = False

        if state.is_terminal():
            # Check that an offer has actually been made
            if state.system_made_offer:
                dialogue_success, reward = self.evaluate_dialog_success(
                    goal.constraints, goal.actual_requests, state
                )

            else:
                reward = self.failure_penalty
                dialogue_success = False
        else:
            reward = self.turn_penalty

        task_success = self.evaluate_task_success(goal, state)

        return reward, dialogue_success, task_success

    def evaluate_task_success(self, goal: Goal, state: SlotFillingDialogueState):
        # Liu & Lane ASRU 2017 Definition of task success
        # We don't care for slots that are not in the goal constraints
        assert goal.ground_truth is None
        if any(filled_wrongly(slot, state, goal) for slot in goal.constraints) or any(
            request_was_not_done(goal, req) for req in goal.requests
        ):
            task_success = False
        else:
            task_success = True
        return task_success

    def evaluate_dialog_success(self, goal_constraints, goal_actual_requests, state):
        offered_right_one = self.check_that_offered_item_meets_users_constraints(
            goal_constraints, state.item_in_focus
        )

        def all_requests_met(actual_requests):
            return not any(
                isinstance(act_item.value, list) and len(act_item.value) == 0
                for act_item in actual_requests.values()
            )

        if offered_right_one and all_requests_met(goal_actual_requests):
            reward = self.success_reward
            dialogue_success = True
        else:
            reward = self.failure_penalty
            dialogue_success = False

        return dialogue_success, reward

    def check_that_offered_item_meets_users_constraints(
        self, goal_constraints, item_in_focus
    ):
        offered_right_one = True
        for constr in goal_constraints:

            if item_in_focus:
                if (
                    item_in_focus[constr] != goal_constraints[constr].value
                    and goal_constraints[constr].value != "dontcare"
                ):
                    offered_right_one = False
                    break
        return offered_right_one
