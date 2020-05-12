import inspect
import os
from typing import Dict, List

import seaborn as sns
import pandas as pd
from util import data_io
from matplotlib import pyplot as plt


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
    num_runs = get_num_runs(data)

    filter_funs = [
        lambda d: d['exp_name'].startswith('pytorch_a2c'),
        lambda d: d['exp_name'].endswith('error_sim'),
        lambda d: d['exp_name'].endswith('two_slots'),
    ]
    for filter_fun in filter_funs:
        df = pd.DataFrame(data=list(filter(filter_fun,data)))
        fig, ax = plt.subplots(figsize=(30, 10))
        ax = sns.boxplot(
            ax=ax,
            x="exp_name", y="success-rate", hue="split-name", palette=["m", "g", "r"], data=df
        )
        # sns.despine(offset=10, trim=True)
        ax.set_title("%d runs, 2000 train and 1000 eval dialogues" % (num_runs))
        # ax.set_xlabel("")
        filter_fun_str = inspect.getsourcelines(filter_fun)[0][0].split(':')[1].strip('\n').strip(',')
        ax.figure.savefig(figure_file.split('.')[0]+"_%s.png"%filter_fun_str)

    # plt.show()
    plt.close()


def get_num_runs(data):
    exp_names = list(set([d["exp_name"] for d in data]))
    num_runs = len([d for d in data if
                    d["exp_name"] == exp_names[0] and d['split-name'] == 'train'])
    return num_runs


if __name__ == "__main__":
    scoring_runs = list(data_io.read_jsonl(os.environ["HOME"]+"/gunther/DIALOGUE/plato/scores.jsonl"))

    plot_results(scoring_runs)
