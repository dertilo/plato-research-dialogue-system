from typing import List

from Dialogue.Action import DialogueActItem
from Dialogue.State import SlotFillingDialogueState
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal


def filled_wrongly(
    gold_slot: str, state: SlotFillingDialogueState, act_item: DialogueActItem
):
    return (
        user_does_care(act_item)
        and gold_slot_was_filled(gold_slot, state)
        and with_wrong_value(act_item, gold_slot, state)
    )


def user_does_care(act_item):
    return act_item.value != "dontcare"


def with_wrong_value(act_item, gold_slot, state):
    return state.slots_filled[gold_slot] != act_item.value


def gold_slot_was_filled(gold_slot, state):
    return gold_slot in state.slots_filled


def request_was_not_done(act_item):
    return isinstance(act_item.value, list) and len(act_item.value) == 0


def all_requests_met(act_items: List[DialogueActItem]):
    return not any(request_was_not_done(act_item) for act_item in act_items)


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
        if not any(
            filled_wrongly(slot, state, act_item)
            for slot, act_item in goal.constraints.items()
        ) and all_requests_met(goal.requests.values()):
            task_success = True
        else:
            task_success = False
        return task_success

    def evaluate_dialog_success(self, goal_constraints, goal_actual_requests, state):

        if offered_item_meets_all_user_constraints(
            goal_constraints, state.item_in_focus
        ) and all_requests_met(goal_actual_requests.values()):
            reward = self.success_reward
            dialogue_success = True
        else:
            reward = self.failure_penalty
            dialogue_success = False

        return dialogue_success, reward


def offered_item_meets_all_user_constraints(goal_constraints, item_in_focus):
    def wrong_value(constr: str):
        act_item = goal_constraints[constr]
        item_value = act_item.value
        # TODO: here a "meet-constraint"-method is needed to check for other Operation but "equality" -> lowerthan, greaterthan
        return item_in_focus[constr] != item_value and user_does_care(act_item)

    offered_right_one = not any(wrong_value(constr) for constr in goal_constraints)
    return offered_right_one
