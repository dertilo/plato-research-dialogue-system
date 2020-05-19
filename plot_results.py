import inspect
import os
from typing import Dict, List

import seaborn as sns
import pandas as pd
from util import data_io
from matplotlib import pyplot as plt


def plot_results(scoring_runs: List[Dict], save_path):
    sns.set(style="ticks", palette="pastel")
    train_dialogues = scoring_runs[0]["train_dialogues"]
    eval_dialogues = scoring_runs[0]["eval_dialogues"]
    warmup_dialogues = scoring_runs[0]["num_warmup_dialogues"]

    data = [
        {
            "exp_name": run["name"],
            "success-rate": run["scores"][split_name]["success-rate"],
            "split-name": split_name,
        }
        for run in scoring_runs
        for split_name in ["train", "eval"]
    ]
    num_runs = get_num_runs(data)

    filter_funs = [
        lambda d: d["exp_name"].startswith("pytorch_a2c"),
        lambda d: d["exp_name"].endswith("error_sim"),
        lambda d: d["exp_name"].endswith("two_slots"),
        lambda d: any([s in d["exp_name"] for s in ["q_learning", "wolf"]]),
        lambda d: True,
    ]
    for filter_fun in filter_funs:
        df = pd.DataFrame(data=list(filter(filter_fun, data)))
        if df.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(16, 8))
        chart = sns.scatterplot(
            ax=ax,
            y="exp_name",
            x="success-rate",
            hue="split-name",
            palette="deep",
            data=df,
            alpha=0.8,s=100
        )
        # sns.despine(offset=10, trim=True)
        ax.set_title(
            "%d runs, %d train and %d eval dialogues, %d warmup"
            % (num_runs, train_dialogues, eval_dialogues, warmup_dialogues)
        )
        # sns.set()
        sns.set_style("whitegrid")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        filter_fun_str = (
            inspect.getsourcelines(filter_fun)[0][0]
            .split(":")[1]
            .strip("\n")
            .strip(",")
        )
        # plt.gcf().subplots_adjust(bottom=0.15,left=0.15, right=0.15)
        plt.tight_layout()
        ax.figure.savefig("boxplots_%s.png" % filter_fun_str)

    # plt.show()
    plt.close()


def get_num_runs(data):
    exp_names = list(set([d["exp_name"] for d in data]))
    num_runs = len(
        [
            d
            for d in data
            if d["exp_name"] == exp_names[0] and d["split-name"] == "train"
        ]
    )
    return num_runs


if __name__ == "__main__":
    path = os.environ["HOME"] + "/gunther/data/plato_results/40000_4000"
    # path = os.environ["HOME"] + "/gunther/data/plato_results/5000_500_again"
    # file = "scores_2000traindialogues.jsonl"
    scoring_runs = list(data_io.read_jsonl(path + "/results.jsonl"))

    plot_results(scoring_runs, path)
