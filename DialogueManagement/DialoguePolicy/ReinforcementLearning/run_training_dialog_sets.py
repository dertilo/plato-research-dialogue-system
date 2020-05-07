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


def build_config(algo="pytorch_a2c", do_train=True, exploration_decay_rate=0.995):
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
                    "type": algo,
                    "train": do_train,
                    "learning_rate": 0.25,  # not use by policy-based (pytorch) agents
                    "learning_decay_rate": 0.995,
                    "discount_factor": 0.99,
                    "exploration_rate": 1.0,
                    "exploration_decay_rate": exploration_decay_rate,
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
    if hasattr(ca.dialogue_manager.policy,'num_pos'):
        num_pos = ca.dialogue_manager.policy.num_pos
        pbar.postfix[0]["num-pos"] = running_average(
            running_factor, pbar.postfix[0]["num-pos"], num_pos
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
    params_to_monitor = {"dialogue": 0, "success-rate": 0.0, "loss": 0.0,'num-pos':0.0}
    running_factor = np.exp(np.log(0.05) / 100)  # after 100 steps sunk to 0.05
    ca.dialogue_manager.policy.warm_up_mode = True
    with tqdm(postfix=[params_to_monitor]) as pbar:
        for dialogue in range(num_dialogues):
            if dialogue > 100 and ca.dialogue_manager.policy.warm_up_mode:
                ca.dialogue_manager.policy.warm_up_mode = False
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


def train_evaluate(algo, train_dialogues=300, eval_dialogues=1000, exploration_decay_rate=0.995):
    clean_dir("logs")
    clean_dir("policies")
    return {
        "train": run_it(build_config(algo, do_train=True, exploration_decay_rate=exploration_decay_rate), train_dialogues),
        "eval": run_it(build_config(algo, do_train=False), eval_dialogues),
    }


if __name__ == "__main__":


    base_path = "/Users/shillmann/PycharmProjects/alex-plato/experiments/run_it_base_dir" # "."

    chdir("%s" % base_path)
    # algos = ['pytorch_reinforce','pytorch_a2c','q_learning']
    algos = ['q_learning']
    # algos = ['pytorch_a2c']
    # algos = ['pytorch_reinforce']

    # list of tuples: (num. of dialogues, exploration decay rate)
    train_dialogues = [(1000, 0.9958783),
                       (5000, 0.9992472),
                       (10000, 0.9992472)]

    #scores = {k: train_evaluate(k,train_dialogues=1000,eval_dialogues=1000) for k in algos}

    scores = {}
    runs = 3
    to_be_run = len(algos) * len(train_dialogues) * runs
    run_count = 1
    for r in range(runs):
        for a in algos:
            for d in train_dialogues:
                key = '_'.join([a, str(d[0])])
                print('Run {} of {}'.format(run_count, to_be_run))

                if key not in scores:
                    scores[key] = {}

                scores[key][r] = \
                    train_evaluate(a, train_dialogues=d[0], exploration_decay_rate=d[1], eval_dialogues=1000)

                run_count += 1

    pprint(scores)

    '''
    1000it [03:20,  4.98it/s[{'dialogue': 999, 'success-rate': 0.84, 'loss': 24.072, 'num-pos': 7.84, 'eps': 0.00998645168764533}]]
    {'learned': 0, 'random': 0, 'warmup': 0}
    WARNING! SlotFillingDialogueState not provided with slots, using default CamRest slots.
    1000it [02:12,  7.54it/s[{'dialogue': 999, 'success-rate': 0.84, 'loss': 0.0, 'num-pos': 0.0, 'eps': 1.0}]]
    {'learned': 0, 'random': 0, 'warmup': 0}
    
    1000it [03:07,  5.32it/s[{'dialogue': 999, 'success-rate': 0.84, 'loss': 10.094, 'num-pos': 5.16, 'eps': 0.00998645168764533}]]
    {'learned': 0, 'random': 0, 'warmup': 0}
    WARNING! SlotFillingDialogueState not provided with slots, using default CamRest slots.
    1000it [02:06,  7.90it/s[{'dialogue': 999, 'success-rate': 0.84, 'loss': 0.0, 'num-pos': 0.0, 'eps': 1.0}]]
    {'learned': 0, 'random': 0, 'warmup': 0}
    Warning! Q DialoguePolicy file policies/agent not found
    
    1000it [03:03,  5.44it/s[{'dialogue': 999, 'success-rate': 0.83, 'loss': 0.0, 'num-pos': 0.0, 'eps': 0.00998645168764533}]]
    {'learned': 6062, 'random': 1585, 'warmup': 2464}
    1000it [01:54,  8.75it/s[{'dialogue': 999, 'success-rate': 0.84, 'loss': 0.0, 'num-pos': 0.0, 'eps': 0.00998645168764533}]]
    {'learned': 8803, 'random': 537, 'warmup': 0}
    {'pytorch_a2c': {'eval': {'avg-reward': 15.235800000000026,
                              'avg-turns': 12.926,
                              'success-rate': 80.2},
                     'train': {'avg-reward': 13.434699999999989,
                               'avg-turns': 11.9,
                               'success-rate': 71.39999999999999}},
     'pytorch_reinforce': {'eval': {'avg-reward': 19.058649999999957,
                                    'avg-turns': 11.407,
                                    'success-rate': 98.0},
                           'train': {'avg-reward': 18.72715000000002,
                                     'avg-turns': 11.301,
                                     'success-rate': 96.39999999999999}},
     'q_learning': {'eval': {'avg-reward': 15.733949999999988,
                             'avg-turns': 11.423,
                             'success-rate': 82.39999999999999},
                    'train': {'avg-reward': 14.83264999999999,
                              'avg-turns': 12.204,
                              'success-rate': 78.5}}}
    '''