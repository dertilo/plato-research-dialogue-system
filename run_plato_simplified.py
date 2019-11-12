from pprint import pprint
from tqdm import tqdm

from ConversationalAgent.ConversationalSingleAgentSimplified import (
    ConversationalSingleAgent,
)

import yaml
import os.path
import time
import random


def run_single_agent(config, num_dialogues):
    ca = ConversationalSingleAgent(config)
    ca.initialize()

    params_to_monitor = {"dialogue": 0, "success-rate": 0.0, "reward": 0.0}
    running_factor = 0.99
    with tqdm(postfix=[params_to_monitor]) as pbar:

        for dialogue in range(num_dialogues):
            ca.start_dialogue()
            while not ca.terminated():
                ca.continue_dialogue()

            ca.end_dialogue()

            pbar.postfix[0]["dialogue"] = dialogue
            success = int(ca.recorder.dialogues[-1][-1]["success"])
            reward = int(ca.recorder.dialogues[-1][-1]["cumulative_reward"])
            pbar.postfix[0]["success-rate"] = round(
                running_factor * pbar.postfix[0]["success-rate"]
                + (1 - running_factor) * success,
                2,
            )
            pbar.postfix[0]["reward"] = round(
                running_factor * pbar.postfix[0]["reward"]
                + (1 - running_factor) * reward,
                2,
            )
            pbar.update()

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

    return statistics


def arg_parse(cfg_filename: str = "config/train_reinforce.yaml"):
    """
    This function will parse the configuration file that was provided as a
    system argument into a dictionary.

    :return: a dictionary containing the parsed config file.
    """

    random.seed(time.time())

    assert os.path.isfile(cfg_filename)
    assert cfg_filename.endswith(".yaml")

    with open(cfg_filename, "r") as file:
        cfg_parser = yaml.load(file, Loader=yaml.Loader)

    interaction_mode = "simulation"
    num_agents = 1

    dialogues = int(cfg_parser["DIALOGUE"]["num_dialogues"])

    if "interaction_mode" in cfg_parser["GENERAL"]:
        interaction_mode = cfg_parser["GENERAL"]["interaction_mode"]

        if "agents" in cfg_parser["GENERAL"]:
            num_agents = int(cfg_parser["GENERAL"]["agents"])

        elif interaction_mode == "multi_agent":
            print(
                "WARNING! Multi-Agent interaction mode selected but "
                "number of agents is undefined in config."
            )

    return {
        "cfg_parser": cfg_parser,
        "dialogues": dialogues,
        "interaction_mode": interaction_mode,
        "num_agents": num_agents,
        "test_mode": False,
    }


if __name__ == "__main__":
    arguments = arg_parse(
        cfg_filename="../alex-plato/Config/simulate_agenda_dacts_train.yaml"
    )
    statistics = run_single_agent(arguments["cfg_parser"], 100)

    pprint(f"Results:\n{statistics}")
