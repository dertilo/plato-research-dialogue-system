from typing import Dict, List

import seaborn as sns
import pandas as pd
from util import data_io


def plot_results(scoring_runs: List[Dict], figure_file="results.png"):
    sns.set(style="ticks", palette="pastel")
    data = [
        {
            "exp_name": name,
            "success-rate": scores[split_name]["success-rate"],
            "split-name": split_name,
        }
        for run in scoring_runs
        for name, scores in run.items()
        for split_name in ["train", "eval"]
    ]
    df = pd.DataFrame(data=data)
    ax = sns.boxplot(
        x="exp_name", y="success-rate", hue="split-name", palette=["m", "g", "r"], data=df
    )
    # sns.despine(offset=10, trim=True)
    ax.set_title("%d runs, 1000 train and 1000 eval dialogues" % (len(scoring_runs)))
    # ax.set_xlabel("")
    ax.figure.savefig(figure_file)
    from matplotlib import pyplot as plt

    # plt.show()
    plt.close()


if __name__ == "__main__":
    scoring_runs = list(data_io.read_jsonl("/tmp/scores.jsonl"))

    plot_results(scoring_runs)
