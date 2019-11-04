import argparse
import os
import pickle
import matplotlib.pyplot as plt
from numpy import mean, median


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

    fig, axs = plt.subplots(4, 1, constrained_layout=True)
    fig.suptitle('Training and Dialog Parameters (window size: {})'.format(window_size), fontsize=16)

    x_values = range(window_size, len(dialogues)+1)

    # Plot cumulative reward
    rewards = [d[-1]['cumulative_reward'] for d in dialogues]

    ws_rewards = [rewards[i:i + window_size] for i in range(len(rewards) - (window_size - 1))]
    ws_r_means = [mean(x) for x in ws_rewards]

    axs[0].plot(x_values, ws_r_means)
    axs[0].set_title('Cumulative Reward '.format(window_size))
    axs[0].set_xlabel('training dialogs')
    axs[0].set_ylabel('mean reward')

    # Plot dialog length
    lengths = [len(d) for d in dialogues]
    ws_lengths = [lengths[i:i + window_size] for i in
                  range(len(lengths) - (window_size - 1))]
    ws_l_mean = [mean(l) for l in ws_lengths]

    axs[1].plot(x_values, ws_l_mean)
    axs[1].set_title('Dialog Length'.format(window_size))
    axs[1].set_xlabel('training dialogs')
    axs[1].set_ylabel('mean number of turns')

    # plot task success rate
    task_success = [int(d[-1]['task_success']) for d in dialogues]
    ws_task_success = [task_success[i:i + window_size] for i in range(len(task_success) - (window_size - 1))]
    ws_ts_rate = [sum(x)/len(x) for x in ws_task_success]

    axs[2].plot(x_values, ws_ts_rate)
    axs[2].set_title('Task Success'.format(window_size))
    axs[2].set_xlabel('training dialogs')
    axs[2].set_ylabel('task success rate')

    # plot task success rate
    success = [int(d[-1]['success']) for d in dialogues]
    ws_success = [success[i:i + window_size] for i in range(len(success) - (window_size - 1))]
    ws_s_rate = [sum(x) / len(x) for x in ws_success]

    axs[3].plot(x_values, ws_s_rate)
    axs[3].set_title('Success'.format(window_size))
    axs[3].set_xlabel('training dialogs')
    axs[3].set_ylabel('success rate')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('An analyser for PLATO log files')
    parser.add_argument('-f', required=True, type=str, help='Path to the log file.')
    parser.add_argument('-p', required=False, type=int, default=1,
                        help='Plot graphs with given window size (default 1).')
    args = parser.parse_args()

    dialogues = load_file(args.f)
    
    print_basic_information(dialogues)
    
    if 'p' in args:
        window_size = 1
        if args.p is not None:
            window_size = args.p
        plot_cumulative_reward(dialogues, window_size)
