import os
import random
import shutil
import traceback
from copy import copy, deepcopy
from dataclasses import dataclass
from os import chdir
from pprint import pprint
from time import time, sleep
from typing import Dict, Any, NamedTuple

import numpy as np
from tqdm import tqdm
from util import data_io
from util.worker_pool import GenericTask, WorkerPool

from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent
from plot_results import plot_results


def one_dialogue(ca):
    ca.start_dialogue()
    while not ca.terminated():
        ca.continue_dialogue()
    ca.end_dialogue()


def build_config(algo="pytorch_a2c", error_sim=False, two_slots=False):
    return {
        "GENERAL": {
            "print_level": "info",
            "interaction_mode": "simulation",
            "agents": 1,
            "runs": 5,
            "experience_logs": {"save": True, "load": False, "path": None,},
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
                "pop_distribution": [50, 50] if two_slots else [1.0],
                "slot_confuse_prob": 0.05 if error_sim else 0.0,
                "op_confuse_prob": 0.0,
                "value_confuse_prob": 0.05 if error_sim else 0.0,
            },
            "DM": {
                "policy": {
                    "type": algo,
                    "train": False,
                    "learning_rate": 0.25,  # not use by policy-based (pytorch) agents
                    "learning_decay_rate": 0.995,
                    "discount_factor": 0.99,
                    "exploration_rate": 1.0,
                    "exploration_decay_rate": 0.996,
                    "min_exploration_rate": 0.01,
                    "policy_path": None,
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


    results = {
        "success-rate": success_rate,
        "avg-reward": avg_reward,
        "avg-turns": avg_turns,
    }

    if hasattr(ca.dialogue_manager.policy, "counter"):
        results["counter"]=ca.dialogue_manager.policy.counter
    return results


def clean_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


import tempfile

@dataclass
class Experiment:
    job_id: int
    name: str
    config: dict
    train_dialogues: int = 1000
    eval_dialogues: int = 1000
    num_warmup_dialogues:int = 200
    scores:dict=None



def train_evaluate(job: Experiment, LOGS_DIR)->Dict:
    # log_dir = tempfile.mkdtemp(suffix="logs")
    policies_dir = tempfile.mkdtemp(suffix="policies")
    exp_name = job.name + "_" + str(job.job_id)

    job.config["AGENT_0"]["DM"]["policy"]["policy_path"] = "%s/agent" % policies_dir
    train_config = deepcopy(job.config)
    train_config["GENERAL"]["experience_logs"]["path"] = (
        LOGS_DIR + "/" + exp_name + "_train.pkl"
    )
    train_config["AGENT_0"]["DM"]["policy"]["train"] = True
    eval_config = deepcopy(job.config)
    eval_config["GENERAL"]["experience_logs"]["path"] = (
        LOGS_DIR + "/" + exp_name + "_eval.pkl"
    )

    return {
        "train": run_it(
            train_config,
            job.train_dialogues,
            num_warmup_dialogues=job.num_warmup_dialogues,
            use_progress_bar=False,
        ),
        "eval": run_it(eval_config, job.eval_dialogues, use_progress_bar=False),
    }


class PlatoScoreTask(GenericTask):

    @staticmethod
    def build_task_data(**task_params) -> Dict[str, Any]:
        return task_params

    @classmethod
    def process(cls, job: Experiment, task_data: Dict[str, Any]):
        for retry in range(15):
            import torch
            mems = [torch.cuda.memory_allocated(device=i) for i in [0, 1]]
            # device_id = np.argmin(mems)
            random.seed(job.job_id*random.randint(0,9999))
            device_id = random.randint(0,1)
            print("USING GPU: %d; mems: %s"%(device_id,str(mems)))
            with torch.cuda.device(device_id):
                try:
                    job.scores = train_evaluate(job,task_data["LOGS_DIR"])
                    break
                except Exception as e:
                    # traceback.print_exc()
                    pass
            sleep(5)
        return job.__dict__


def build_name(algo, error_sim, two_slots):
    name = [algo]
    name += ["error_sim"] if error_sim else []
    name += ["two_slots"] if two_slots else []
    return "_".join(name)


eid=[0]
def get_id():
    global eid
    eid[0]+=1
    return eid[0]

def multi_eval(algos,LOGS_DIR, num_eval=5, num_workers=12):

    """
    evaluating 12 jobs with 1 workers took: 415.78 seconds
    evaluating 12 jobs with 3 workers took: 154.78 seconds
    evaluating 12 jobs with 6 workers took: 91.88 seconds
    evaluating 12 jobs with 12 workers took: 70.68 seconds

    on gunther one gets cuda out of mem error with num_workers>12
    """

    task = PlatoScoreTask(LOGS_DIR = LOGS_DIR)


    jobs = [
        Experiment(
            job_id=get_id(),
            name=build_name(algo, error_sim, two_slots),
            config=build_config(algo, error_sim=error_sim, two_slots=two_slots),
            train_dialogues=warmupd*10,
            eval_dialogues=1000,
            num_warmup_dialogues=warmupd
        )
        for _ in range(num_eval)
        for error_sim in [False,True]
        for two_slots in [False,True]
        for warmupd in [500,4000]
        for algo in algos
    ]
    start = time()

    outfile = LOGS_DIR+"/results.jsonl"

    mode = "wb"
    if os.path.isdir(LOGS_DIR):
        results = list(data_io.read_jsonl(outfile))
        done_ids = [e['job_id'] for e in results]
        jobs = [e for e in jobs if e.job_id not in done_ids]
        print('only got %d jobs to do'%len(jobs))
        print([e.job_id for e in jobs])
        mode = "ab"
    else:
        os.makedirs(LOGS_DIR)

    if num_workers > 0:
        num_workers = min(len(jobs),num_workers)
        with WorkerPool(processes=num_workers, task=task, daemons=False) as p:
            processed_jobs = p.process_unordered(jobs)
            data_io.write_jsonl(outfile, processed_jobs, mode=mode)
    else:
        with task as t:
            processed_jobs = [t(job) for job in jobs]
            data_io.write_jsonl(outfile, processed_jobs, mode=mode)

    scoring_runs = list(data_io.read_jsonl(outfile))
    plot_results(scoring_runs,LOGS_DIR)

    print(
        "evaluating %d jobs with %d workers took: %0.2f seconds"
        % (len(jobs), num_workers, time() - start)
    )


if __name__ == "__main__":
    LOGS_DIR = "plato_results"
    # clean_dir(LOGS_DIR)

    algos = ["pytorch_a2c", "pytorch_reinforce", "q_learning", "wolf_phc"]
    multi_eval(algos,LOGS_DIR, num_workers=4,num_eval=1)
    # algo = "pytorch_reinforce"
    # error_sim = False
    # two_slots = True
    # train_evaluate(
    #     Job(
    #         name=build_name(algo, error_sim, two_slots),
    #         config=build_config(algo, error_sim=error_sim, two_slots=two_slots),
    #         train_dialogues=200,
    #         eval_dialogues=100,
    #     )
    # )
    # scores = {k: train_evaluate(k,train_dialogues=1000,eval_dialogues=1000) for k in algos}
    # pprint(scores)

    """
    evaluating 12 jobs with 6 workers took: 801.11 seconds
    """
