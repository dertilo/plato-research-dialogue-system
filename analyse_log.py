import argparse
import os
import pickle
import matplotlib.pyplot as plt
from numpy import mean


def load_file(path):
    dialogues = []
    if isinstance(path, str) and os.path.isfile(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
            if 'dialogues' in obj:
                dialogues = obj['dialogues']

    return dialogues


def print_basic_information(dialogues: list):
    print('Number of dialogues: {}'.format(len(dialogues)))


def plot_cumulative_reward(dialogues, window_size=1):
    rewards = [d[-1]['cumulative_reward'] for d in dialogues]

    ws_rewards = [rewards[i:i+window_size] for i in range(len(rewards)-(window_size-1))]
    ws_means = [mean(x) for x in ws_rewards]

    plt.plot(ws_means)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('An analyser for PLATO log files')
    parser.add_argument('-f', required=True, type=str, help='Path to the log file.')
    parser.add_argument('-p', required=False, type=int, default=1, help='Plot cumulative reward with given window size (default 1).')
    args = parser.parse_args()

    dialogues = load_file(args.f)
    
    print_basic_information(dialogues)
    
    if 'p' in args:
        window_size = 1
        if args.p is not None:
            window_size = args.p
        plot_cumulative_reward(dialogues, window_size)
