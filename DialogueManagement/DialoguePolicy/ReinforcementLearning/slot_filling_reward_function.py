from typing import List, Dict

from Dialogue.Action import DialogueActItem
from Dialogue.State import SlotFillingDialogueState
from UserSimulator.AgendaBasedUserSimulator.Goal import Goal


def user_does_care(act_item):
    return act_item.value != "dontcare"


def request_was_not_done(value):
    return isinstance(value, list) and len(value) == 0


def all_requests_done(act_items: List[DialogueActItem]):
    return all(not request_was_not_done(act_item.value) for act_item in act_items)


class SlotFillingReward(object):
    def __init__(self):
        self.goal = None
        self.turn_penalty = -0.05
        self.failure_penalty = -1
        self.success_reward = 20

    def calculate(self, state: SlotFillingDialogueState, goal: Goal):

        assert goal is not None

        dialogue_success, reward = self.evaluate_dialogue_success(goal, state)

        return reward, dialogue_success

    def evaluate_dialogue_success(self, goal, state):
        if state.is_terminal():
            # Check that an offer has actually been made
            if state.system_made_offer:
                dialogue_success, reward = self._evaluate_offer(
                    goal.constraints, list(goal.requests_made.values()), state
                )

            else:
                reward = self.failure_penalty
                dialogue_success = False
        else:
            reward = self.turn_penalty
            dialogue_success = False
        return dialogue_success, reward

    def _evaluate_offer(
        self,
        goal_constraints,
        requests_made_by_user: List[DialogueActItem],
        state: SlotFillingDialogueState,
    ):

        if offered_item_meets_all_user_constraints(
            goal_constraints, state.item_in_focus
        ) and all_requests_done(requests_made_by_user):
            reward = self.success_reward
            dialogue_success = True
        else:
            reward = self.failure_penalty
            dialogue_success = False

        return dialogue_success, reward


def offered_item_meets_all_user_constraints(
    goal_constraints: Dict[str, DialogueActItem], item_in_focus
):
    def wrong_value(value, constr_act_item: DialogueActItem):
        # TODO: here a "meet-constraint"-method is needed to check for other Operation but "equality" -> lowerthan, greaterthan
        return user_does_care(constr_act_item) and value != constr_act_item.value

    offered_right_one = not any(
        wrong_value(item_in_focus[slot_name], dialog_act)
        for slot_name, dialog_act in goal_constraints.items()
    )
    return offered_right_one
