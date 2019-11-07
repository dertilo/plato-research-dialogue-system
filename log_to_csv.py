import argparse
import pickle
from os import path
import csv


def convert(log, csv_p):

    # read input (pickle)
    print('Read input.')
    with open(log, 'rb') as file:
        log_data = pickle.load(file)
    dialogues = log_data['dialogues']

    if len(dialogues) < 1:
        print('Could not read any dialogues from input log file.')
        return

    # extracted needed information
    print('Extract information.')
    csv_data = list()
    for dialog in dialogues:
        last_turn = dialog[-1]
        d = dict()
        d['cumulative_reward'] = last_turn['cumulative_reward']
        d['success'] = int(last_turn['success'])
        d['task_success'] = int(last_turn['task_success'])
        d['length'] = len(dialog)

        csv_data.append(d)

    print('Write output')
    # write output (csv)
    with open(csv_p, 'w', newline='') as csv_file:
        fieldnames = csv_data[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('An analyser for PLATO log files')
    parser.add_argument('-i', required=True, type=str, help='Input log file (pickle).')
    parser.add_argument('-o', required=True, type=str, help='Output csv file.')
    parser.add_argument('-f', required=False, action='store_true', help='Fore overwriting of an existing output file.')
    args = parser.parse_args()

    force_overwriting = args.f

    # check if input file exists
    log_path = args.i
    if not path.exists(log_path):
        print('Input file path ({}) does not exist!'.format(log_path))
        exit(-1)
    if not path.isfile(log_path):
        print('Input ({}) must be a file and  not e.g. a directory!'.format(log_path))
        exit(-1)

    csv_path = args.o
    if path.exists(csv_path) and not force_overwriting:
        print('Output file path ({}) already exist! Use -f to force overwriting.'.format(csv_path))
        exit(-1)
    if path.isdir(log_path):
        print('Output ({}) is a directory! '.format(log_path))
        exit(-1)

    convert(log_path, csv_path)
