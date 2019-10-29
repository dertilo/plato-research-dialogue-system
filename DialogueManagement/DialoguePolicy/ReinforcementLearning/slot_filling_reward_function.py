"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from abc import ABC, abstractmethod
from copy import deepcopy



class SlotFillingReward(object):
    def __init__(self):
        """
        Set default values for turn penalty, success, and failure.
        """

        self.goal = None

        self.turn_penalty = -0.05
        self.failure_penalty = -1
        self.success_reward = 20


    def calculate(self, state, actions, goal=None, force_terminal=False,
                  agent_role='system'):
        """
        Calculate the reward to be assigned for taking action from state.


        :param state: the current state
        :param actions: the action taken from the current state
        :param goal: the agent's goal, used to assess success
        :param force_terminal: force state to be terminal
        :param agent_role: the role of the agent
        :return: a number, representing the calculated reward
        """

        reward = self.turn_penalty

        if goal is None:
            print('Warning: SlotFillingReward() called without a goal.')
            return 0, False, False

        else:
            dialogue_success = False

            if state.is_terminal() or force_terminal:
                # Check that an offer has actually been made
                if state.system_made_offer:
                    dialogue_success = True

                    # Check that the item offered meets the user's constraints
                    for constr in goal.constraints:
                        if goal.ground_truth:
                            # Multi-agent case
                            if goal.ground_truth[constr] != \
                                    goal.constraints[constr].value and \
                                    goal.constraints[constr].value != \
                                    'dontcare':
                                reward += self.failure_penalty
                                dialogue_success = False
                                break

                        elif state.item_in_focus:
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

                else:
                    reward += self.failure_penalty
                    dialogue_success = False

        # Liu & Lane ASRU 2017 Definition of task success
        task_success = None
        if agent_role == 'system':
            task_success = True
            # We don't care for slots that are not in the goal constraints
            for slot in goal.constraints:
                # If the system proactively informs about a slot the user has
                # not yet put a constraint upon,
                # the user's DState is updated accordingly and the user would
                # not need to put that constraint.
                if goal.ground_truth:
                    if goal.ground_truth[slot] != \
                            goal.constraints[slot].value and \
                            goal.constraints[slot].value != 'dontcare':
                        task_success = False
                        break

                # Fall back to the noisier signal, that is the slots filled.
                elif slot in state.slots_filled and \
                        state.slots_filled[slot] != \
                        goal.constraints[slot].value and \
                        goal.constraints[slot].value != 'dontcare':
                    task_success = False
                    break

            for req in goal.requests:
                if not goal.requests[req].value:
                    task_success = False
                    break

        return reward, dialogue_success, task_success