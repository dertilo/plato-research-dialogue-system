import os
import shutil
from os import chdir
from pprint import pprint
from time import time
from typing import Dict, Any, NamedTuple

import numpy as np
from tqdm import tqdm
from util import data_io
from util.worker_pool import GenericTask, WorkerPool

from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent


def one_dialogue(ca):
    ca.start_dialogue()
    while not ca.terminated():
        ca.continue_dialogue()
    ca.end_dialogue()


def build_config(log_dir, policies_dir, algo="pytorch_a2c", do_train=True):
    return {
        "GENERAL": {
            "print_level": "info",
            "interaction_mode": "simulation",
            "agents": 1,
            "runs": 5,
            "experience_logs": {
                "save": False,
                "load": False,
                "path": "%s/train_reinforce_logs.pkl" % log_dir,
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
                # "slot_confuse_prob": 0.0,
                # "op_confuse_prob": 0.0,
                # "value_confuse_prob": 0.0,
                # "pop_distribution": [50, 50],
                "slot_confuse_prob": 0.05,
                "op_confuse_prob": 0.0,
                "value_confuse_prob": 0.05,
            },
            "DM": {
                "policy": {
                    "type": algo,
                    "train": do_train,
                    "learning_rate": 0.25,  # not use by policy-based (pytorch) agents
                    "learning_decay_rate": 0.995,
                    "discount_factor": 0.99,
                    "exploration_rate": 1.0,
                    "exploration_decay_rate": 0.996,
                    "min_exploration_rate": 0.01,
                    "policy_path": "%s/agent" % policies_dir,
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
    if hasattr(ca.dialogue_manager.policy, "num_pos"):
        num_pos = ca.dialogue_manager.policy.num_pos
        pbar.postfix[0]["num-pos"] = running_average(
            running_factor, pbar.postfix[0]["num-pos"], num_pos
        )

    eps = ca.dialogue_manager.policy.epsilon
    pbar.postfix[0]["eps"] = eps
    pbar.update()


def run_it(config, num_dialogues=100, num_warmup_dialogues=100, use_progress_bar=True):
    ca = ConversationalSingleAgent(config)
    ca.initialize()
    # if config["AGENT_0"]["DM"]["policy"]["train"] and hasattr(
    #     ca.dialogue_manager.policy, "agent"
    # ):
    #     print(ca.dialogue_manager.policy.agent)
    ca.minibatch_length = 8
    ca.train_epochs = 10
    ca.train_interval = 8
    params_to_monitor = {
        "dialogue": 0,
        "success-rate": 0.0,
        "loss": 0.0,
        "num-pos": 0.0,
    }
    running_factor = np.exp(np.log(0.05) / 100)  # after 100 steps sunk to 0.05
    ca.dialogue_manager.policy.warm_up_mode = True

    class Dummy:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    pbar_supplier = (
        lambda: tqdm(postfix=[params_to_monitor]) if use_progress_bar else Dummy()
    )
    with pbar_supplier() as pbar:
        for dialogue in range(num_dialogues):
            if (
                dialogue > num_warmup_dialogues
                and ca.dialogue_manager.policy.warm_up_mode
            ):
                ca.dialogue_manager.policy.warm_up_mode = False
            one_dialogue(ca)
            if use_progress_bar:
                update_progress_bar(ca, dialogue, pbar, running_factor)

    if hasattr(ca.dialogue_manager.policy, "counter"):
        pprint(ca.dialogue_manager.policy.counter)

    success_rate = 100 * float(ca.num_successful_dialogues / num_dialogues)
    avg_reward = ca.cumulative_rewards / num_dialogues
    avg_turns = float(ca.total_dialogue_turns / num_dialogues)

    return {
        "success-rate": success_rate,
        "avg-reward": avg_reward,
        "avg-turns": avg_turns,
    }


def clean_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


import tempfile


class Job(NamedTuple):
    algo: str
    train_dialogues: int = 1000
    eval_dialogues: int = 1000


def train_evaluate(job: Job):
    log_dir = tempfile.mkdtemp(suffix="logs")
    policies_dir = tempfile.mkdtemp(suffix="policies")
    return {
        "train": run_it(
            build_config(log_dir, policies_dir, job.algo, do_train=True),
            job.train_dialogues,
            num_warmup_dialogues=50,
            use_progress_bar=False,
        ),
        "eval": run_it(
            build_config(log_dir, policies_dir, job.algo, do_train=False),
            job.eval_dialogues,
            use_progress_bar=False,
        ),
    }


class PlatoScoreTask(GenericTask):
    @classmethod
    def process(cls, job: Job, task_data: Dict[str, Any]):
        return train_evaluate(job)


def multi_eval(algos, num_eval=2):

    task = PlatoScoreTask()
    jobs = [
        Job(algo, train_dialogues=100, eval_dialogues=100) for algo in algos
    ] * num_eval
    num_workers = 1
    start = time()

    with WorkerPool(processes=num_workers, task=task, daemons=False) as p:
        results_g = p.process_unordered(jobs)
        data_io.write_jsonl("scores.jsonl", results_g)

    print("evaluation with %d workers took: %0.2f seconds"% (num_workers,time()-start))


if __name__ == "__main__":

    base_path = "."

    chdir("%s" % base_path)
    algos = ["pytorch_a2c", "pytorch_reinforce", "q_learning", "wolf_phc"]
    # algos = ['wolf_phc']
    # algos = ['pytorch_reinforce']
    multi_eval(algos)
    # scores = {k: train_evaluate(k,train_dialogues=1000,eval_dialogues=1000) for k in algos}
    # pprint(scores)

    """
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
    """
