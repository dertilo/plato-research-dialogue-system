import os
import shutil

from runPlatoRDS import run_controller


def clean_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


clean_dir("logs")
clean_dir("policies")
# if os.path.isfile("/tmp/agent"):
#     os.remove("/tmp/agent")

config_file = "train_q_learning.yaml"
# config_file = 'train_dl_reinforce.yaml'
arguments = {
    "cfg_parser": {
        "AGENT_0": {
            "DM": {
                "policy": {
                    "discount_factor": 0.8,
                    "exploration_decay_rate": 0.9,
                    "exploration_rate": 1.0,
                    "learning_decay_rate": 0.995,
                    "learning_rate": 0.25,
                    "min_exploration_rate": 0.01,
                    "policy_path": "policies/q_learning_policy.pkl",
                    "train": True,
                    "type": "q_learning",
                }
            },
            "DST": {"dst": "dummy"},
            "NLG": None,
            "NLU": None,
            "USER_SIMULATOR": {
                "op_confuse_prob": 0.0,
                "patience": 5,
                "pop_distribution": [1.0],
                "simulator": "agenda",
                "slot_confuse_prob": 0.04,
                "value_confuse_prob": 0.04,
            },
            "role": "system",
        },
        "DIALOGUE": {
            "cache_sql_results": True,
            "db_path": "domain/alex-dbase.db",
            "db_type": "sql",
            "domain": "CamRest",
            "initiative": "system",
            "num_dialogues": 40000,
            "ontology_path": "domain/alex-rules.json",
        },
        "GENERAL": {
            "agents": 1,
            "experience_logs": {
                "load": False,
                "path": "logs/train_q_learning_log.pkl",
                "save": True,
            },
            "interaction_mode": "simulation",
            "print_level": "info",
            "runs": 1,
        },
    },
    "dialogues": 200,
    "interaction_mode": "simulation",
    "num_agents": 1,
    "test_mode": False,
    "tests": 1,
}
run_controller(arguments)

eval_args = {
    "cfg_parser": {
        "AGENT_0": {
            "DM": {
                "policy": {
                    "policy_path": "policies/q_learning_policy.pkl",
                    "train": False,
                    "type": "q_learning",
                }
            },
            "DST": {"dst": "dummy"},
            "NLG": None,
            "NLU": None,
            "USER_SIMULATOR": {
                "op_confuse_prob": 0.0,
                "patience": 5,
                "pop_distribution": [1.0],
                "simulator": "agenda",
                "slot_confuse_prob": 0.0,
                "value_confuse_prob": 0.0,
            },
            "role": "system",
        },
        "DIALOGUE": {
            "cache_sql_results": True,
            "db_path": "domain/alex-dbase.db",
            "db_type": "sql",
            "domain": "CamRest",
            "initiative": "system",
            "num_dialogues": 1000,
            "ontology_path": "domain/alex-rules.json",
        },
        "GENERAL": {
            "agents": 1,
            "experience_logs": {
                "load": False,
                "path": "logs/eval_q_learning_log.pkl",
                "save": True,
            },
            "interaction_mode": "simulation",
            "print_level": "info",
            "runs": 1,
        },
    },
    "dialogues": 100,
    "interaction_mode": "simulation",
    "num_agents": 1,
    "test_mode": False,
    "tests": 1,
}
run_controller(eval_args)

