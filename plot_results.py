import seaborn as sns
import pandas as pd
from util import data_io


if __name__ == "__main__":
    scoring_runs = data_io.read_jsonl("/tmp/scores.jsonl")
    num_cross_val = 5

    sns.set(style="ticks", palette="pastel")
    data = [
        {
            "algo": algo,
            "success-rate": scores[split_name]["success-rate"],
            "split-name": split_name,
        }
        for run in scoring_runs
        for algo, scores in run.items()
        for split_name in ["train", "eval"]
    ]
    df = pd.DataFrame(data=data)
    ax = sns.boxplot(
        x="algo", y="success-rate", hue="split-name", palette=["m", "g", "r"], data=df
    )
    # sns.despine(offset=10, trim=True)
    ax.set_title("%d runs, 1000 train and 1000 eval dialogues" % (num_cross_val))
    # ax.set_xlabel("")
    ax.figure.savefig("results.png")
    from matplotlib import pyplot as plt

    # plt.show()

    plt.close()
