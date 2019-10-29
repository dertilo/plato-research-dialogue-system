import os
import pickle
import numpy as np
import argparse

from DialogueManagement.DialoguePolicy.ReinforcementLearning.QPolicy import QPolicy
from Domain import Ontology, DataBase

detail_limit = 10


ontology = Ontology.Ontology('../alex-plato/Domain/alex-rules.json')
database = DataBase.DataBase('../alex-plato/Domain/alex-dbase.db')
qp = QPolicy(ontology, database, agent_role='system', domain='CamRest')

def analyse(model_path=None):
    if model_path is None:
        print('Path to policy is missing.')

    info = None
    if isinstance(model_path, str) and os.path.isfile(model_path):
        with open(model_path, 'rb') as file:
            obj = pickle.load(file)
            if 'i' in obj:
                info = obj['i']

    if info is not None:
        sums = [sum(list(info[k].values())) for k in info.keys()]
        updates_per_state = dict(zip(info.keys(), sums))
        updates_per_state = dict(sorted(updates_per_state.items(), key=lambda x: x[1]))

        count = 1
        for k, v in updates_per_state.items():
            print('{}. {}: {}'.format(count, k, v))
            count += 1

        print('\n')
        print('Distinct states: {}'.format(len(info)))

        values = list(updates_per_state.values())
        mean = np.mean(values)
        print('Mean updates: {}'.format(mean))

        sd = np.std(values)
        print('SD updates: {}'.format(sd))

        median = np.median(values)
        print('Median updates: {}'.format(median))

    else:
        print('Found no info in model file.')

    print('\n')
    print('Actions')
    for k in list(updates_per_state.keys())[-detail_limit:]:
        print('\n')
        print('{}'.format(k))

        for action_enc, action_count in info[k].items():
            action_dec = qp.decode_action(action_enc, system=True)[0]  # TODO: system=True here always correct?
            print('{}: {}'.format(action_dec.__str__(), action_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('An analyser for PLATO model files')
    parser.add_argument('-m', required=True, type=str, help='Path to the pickle model file.')
    args = parser.parse_args()

    path = args.m
    analyse(path)
