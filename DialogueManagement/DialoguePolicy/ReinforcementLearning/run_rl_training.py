import os
import shutil
from os import chdir
import numpy as np
from tqdm import tqdm
from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent


def one_dialogue(ca):
    ca.start_dialogue()
    while not ca.terminated():
        ca.continue_dialogue()
    ca.end_dialogue()


def build_config(do_train=True):
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
                    "type": "pytorch_reinforce",
                    "train": do_train,
                    "learning_rate": 0.01,
                    "learning_decay_rate": 0.995,
                    "discount_factor": 0.8,
                    "exploration_rate": 1.0,
                    "exploration_decay_rate": 1.0,
                    "min_exploration_rate": 0.01,
                    "policy_path": "/tmp/agent",
                }
            },
            "NLU": None,
            "DST": {"dst": "dummy"},
            "NLG": None,
        },
    }


def update_progress_bar(ca: ConversationalSingleAgent, dialogue, pbar, running_factor):

    def running_average(running_factor, old_val, new_val, num_deci=2):
        val = running_factor * old_val + (1 - running_factor) * new_val
        return round(val, num_deci)

    if len(ca.dialogue_manager.policy.losses) > 0:
        loss = ca.dialogue_manager.policy.losses[-1]
    else:
        loss = 0
    pbar.postfix[0]["loss"] = running_average(
        running_factor, pbar.postfix[0]["loss"], loss,3
    )
    pbar.postfix[0]["dialogue"] = dialogue
    success = int(ca.recorder.dialogues[-1][-1]["success"])
    pbar.postfix[0]["success-rate"] = running_average(
        running_factor, pbar.postfix[0]["success-rate"], success
    )

    eps = ca.dialogue_manager.policy.epsilon
    pbar.postfix[0]["eps"] = eps
    pbar.update()


def run_it(config,num_dialogues=100):
    ca = ConversationalSingleAgent(config)
    if config["AGENT_0"]["DM"]["policy"]["train"]:
        print(ca.dialogue_manager.policy.agent)
    ca.initialize()
    ca.minibatch_length = 8
    ca.train_epochs = 10
    ca.train_interval = 8
    params_to_monitor = {"dialogue": 0, "success-rate": 0.0,'loss':0.0}
    running_factor = np.exp(np.log(0.05) / 100)  # after 100 steps sunk to 0.05
    with tqdm(postfix=[params_to_monitor]) as pbar:
        for dialogue in range(num_dialogues):
            one_dialogue(ca)
            update_progress_bar(ca, dialogue, pbar, running_factor)
    # Collect statistics
    statistics = {"AGENT_0": {}}
    statistics["AGENT_0"]["dialogue_success_percentage"] = 100 * float(
        ca.num_successful_dialogues / num_dialogues
    )
    statistics["AGENT_0"]["avg_cumulative_rewards"] = float(
        ca.cumulative_rewards / num_dialogues
    )
    statistics["AGENT_0"]["avg_turns"] = float(ca.total_dialogue_turns / num_dialogues)
    statistics["AGENT_0"]["objective_task_completion_percentage"] = 100 * float(
        ca.num_task_success / num_dialogues
    )
    print(
        "\n\nDialogue Success Rate: {0}\nAverage Cumulative Reward: {1}"
        "\nAverage Turns: {2}".format(
            statistics["AGENT_0"]["dialogue_success_percentage"],
            statistics["AGENT_0"]["avg_cumulative_rewards"],
            statistics["AGENT_0"]["avg_turns"],
        )
    )


if __name__ == "__main__":
    """
    simple test
    """

    def clean_dir(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)

    base_path = "../alex-plato/experiments/exp_09"

    chdir("%s" % base_path)

    clean_dir("logs")
    clean_dir("policies")
    if os.path.isfile("/tmp/agent"):
        os.remove("/tmp/agent")

    config = build_config(do_train=True)
    run_it(config,1000)
    config = build_config(do_train=False)
    run_it(config)
