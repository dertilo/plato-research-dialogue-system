import os
import shutil
from os import chdir
from pprint import pprint

import numpy as np
from tqdm import tqdm
from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent


def one_dialogue(ca):
    ca.start_dialogue()
    while not ca.terminated():
        ca.continue_dialogue()
    ca.end_dialogue()


def build_config(algo="pytorch_a2c", do_train=True):
    return {
        "GENERAL": {
            "print_level": "info",
            "interaction_mode": "simulation",
            "agents": 1,
            "runs": 5,
            "experience_logs": {
                "save": False,
                "load": False,
                "path": "logs/train_reinforce_logs.pkl",
            },
        },
        "DIALOGUE": {
            "num_dialogues": 1000,
            "initiative": "system",
            "domain": "CamRest",
            "ontology_path": "domain/alex-rules.json",
            "db_path": "domain/alex-dbase.db",
            "db_type": "sql",
            "cache_sql_results": True,
        },
        "AGENT_0": {
            "role": "system",
            "USER_SIMULATOR": {
                "simulator": "agenda",
                "patience": 5,
                "pop_distribution": [1.0],
                "slot_confuse_prob": 0.0,
                "op_confuse_prob": 0.0,
                "value_confuse_prob": 0.0,
            },
            "DM": {
                "policy": {
                    # "type": "q_learning",
                    "type": algo,
                    "train": do_train,
                    "learning_rate": 0.25,  # not use by policy-based (pytorch) agents
                    "learning_decay_rate": 0.995,
                    "discount_factor": 0.99,
                    "exploration_rate": 1.0,
                    "exploration_decay_rate": 1.0,
                    "min_exploration_rate": 0.01,
                    "policy_path": "policies/agent",
                }
            },
        },
    }


def update_progress_bar(ca: ConversationalSingleAgent, dialogue, pbar, running_factor):
    def running_average(running_factor, old_val, new_val, num_deci=2):
        val = running_factor * old_val + (1 - running_factor) * new_val
        return round(val, num_deci)

    if hasattr(ca.dialogue_manager.policy, "losses"):
        if len(ca.dialogue_manager.policy.losses) > 0:
            loss = ca.dialogue_manager.policy.losses[-1]
        else:
            loss = 0
        pbar.postfix[0]["loss"] = running_average(
            running_factor, pbar.postfix[0]["loss"], loss, 3
        )
    pbar.postfix[0]["dialogue"] = dialogue
    success = int(ca.recorder.dialogues[-1][-1]["success"])
    pbar.postfix[0]["success-rate"] = running_average(
        running_factor, pbar.postfix[0]["success-rate"], success
    )

    eps = ca.dialogue_manager.policy.epsilon
    pbar.postfix[0]["eps"] = eps
    pbar.update()


def run_it(config, num_dialogues=100, verbose=False):
    ca = ConversationalSingleAgent(config)
    ca.initialize()
    if config["AGENT_0"]["DM"]["policy"]["train"] and hasattr(
        ca.dialogue_manager.policy, "agent"
    ):
        print(ca.dialogue_manager.policy.agent)
    ca.minibatch_length = 8
    ca.train_epochs = 10
    ca.train_interval = 8
    params_to_monitor = {"dialogue": 0, "success-rate": 0.0, "loss": 0.0}
    running_factor = np.exp(np.log(0.05) / 100)  # after 100 steps sunk to 0.05
    with tqdm(postfix=[params_to_monitor]) as pbar:
        for dialogue in range(num_dialogues):
            one_dialogue(ca)
            update_progress_bar(ca, dialogue, pbar, running_factor)

    if hasattr(ca.dialogue_manager.policy, "counter"):
        pprint(ca.dialogue_manager.policy.counter)

    success_rate = 100 * float(ca.num_successful_dialogues / num_dialogues)
    avg_reward = ca.cumulative_rewards / num_dialogues
    avg_turns = float(ca.total_dialogue_turns / num_dialogues)
    if verbose:
        print(
            "\n\nDialogue Success Rate: {0}\nAverage Cumulative Reward: {1}"
            "\nAverage Turns: {2}".format(success_rate, avg_reward, avg_turns,)
        )

    return {
        "success-rate": success_rate,
        "avg-reward": avg_reward,
        "avg-turns": avg_turns,
    }

def clean_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

def train_evaluate(algo, train_dialogues=300,eval_dialogues = 100):
    clean_dir("logs")
    clean_dir("policies")
    return {
        "train": run_it(build_config(algo, do_train=True), train_dialogues),
        "eval": run_it(build_config(algo, do_train=False), eval_dialogues),
    }


if __name__ == "__main__":


    base_path = "."

    chdir("%s" % base_path)
    algos = ['pytorch_reinforce','pytorch_a2c','q_learning']
    scores = {k: train_evaluate(k,train_dialogues=200) for k in algos}
    pprint(scores)