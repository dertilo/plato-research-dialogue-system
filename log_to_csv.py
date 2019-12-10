import argparse
import pickle
from os import path
import csv


def convert_on_turn_level(log, csv_p):
    print('Read input.')
    with open(log, 'rb') as file:
        log_data = pickle.load(file)
    dialogues = log_data['dialogues']

    if len(dialogues) < 1:
        print('Could not read any dialogues from input log file.')
        return

    # extracted needed information
    print('Extract information on turn-level.')
    csv_data = list()
    dialog_id = 0
    for dialog in dialogues:
        turn_id = 0
        for t in dialog:
            turn = dict()
            turn['dialog_id'] = dialog_id
            turn['turn_id'] = turn_id
            turn['cumulative_reward'] = round(t['cumulative_reward'], 2)
            turn['reward'] = t['reward']

            # collect slot names and related values for users's action
            user_slots = list()
            user_values = list()
            user_intent = 'EMPTY'
            if t['state'].user_acts is not None:
                user_act = t['state'].user_acts[0]
                user_intent = user_act.intent
                for p in user_act.params:
                    user_slots.append(p.slot)
                    user_values.append(p.value)
            turn['user_intent'] = user_intent
            turn['user_slots'] = ' '.join(user_slots)
            if len(user_values) > 0 and isinstance(user_values[0], list):
                turn['user_values'] = 'EMPTY_LIST'
            else:
                turn['user_values'] = ' '.join(user_values)

            # collect slot names and related values for system's action
            sys_act = t['action'][0]
            turn['system_intent'] = sys_act.intent
            system_slots = list()
            system_values = list()
            for p in sys_act.params:
                system_slots.append(p.slot)
                system_values.append(p.value)
            turn['system_slots'] = ' '.join(system_slots)
            turn['system_values'] = ' '.join(system_values)

            csv_data.append(turn)

            turn_id += 1

        dialog_id += 1

    print('Write output')
    # write output (csv)
    with open(csv_p, 'w', newline='') as csv_file:
        fieldnames = csv_data[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)

def convert_on_dialog_level(log, csv_p):

    # read input (pickle)
    print('Read input.')
    with open(log, 'rb') as file:
        log_data = pickle.load(file)
    dialogues = log_data['dialogues']

    if len(dialogues) < 1:
        print('Could not read any dialogues from input log file.')
        return

    # extracted needed information
    print('Extract information on dialog-level.')
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
    parser.add_argument('-l', required=False, type=str, default='d',
                        help='Extraction level (d: dialog-level, t: turn-level')
    args = parser.parse_args()

    force_overwriting = args.f
    level = args.l

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

    if level == 'd':
        convert_on_dialog_level(log_path, csv_path)
    elif level == 't':
        convert_on_turn_level(log_path, csv_path)
    else:
        print('Level "{}" is not known.'.format(level))
