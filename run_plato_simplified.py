"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""
from pprint import pprint
from tqdm import tqdm

from ConversationalAgent.ConversationalSingleAgent import ConversationalSingleAgent

__author__ = "Alexandros Papangelis"

import configparser
import yaml
import sys
import os.path
import time
import random

def run_single_agent(config, num_dialogues):
    """
    This function will create an agent and orchestrate the conversation.

    :param config: a dictionary containing settings
    :param num_dialogues: how many dialogues to run for
    :return: some statistics
    """
    ca = ConversationalSingleAgent(config)
    ca.initialize()

    params_to_monitor = {'dialogue': 0, 'success-rate': 0.0}
    running_factor=.99
    with tqdm(postfix=[params_to_monitor]) as pbar:

        for dialogue in range(num_dialogues):
            ca.start_dialogue()
            while not ca.terminated():
                ca.continue_dialogue()

            ca.end_dialogue()

            pbar.postfix[0]['dialogue'] = dialogue
            success = int(ca.recorder.dialogues[-1][-1]['success'])
            pbar.postfix[0]['success-rate'] = round(
                running_factor * pbar.postfix[0]['success-rate'] + (
                            1 - running_factor) * success, 2)
            pbar.update()

    # Collect statistics
    statistics = {'AGENT_0': {}}

    statistics['AGENT_0']['dialogue_success_percentage'] = \
        100 * float(ca.num_successful_dialogues / num_dialogues)
    statistics['AGENT_0']['avg_cumulative_rewards'] = \
        float(ca.cumulative_rewards / num_dialogues)
    statistics['AGENT_0']['avg_turns'] = \
        float(ca.total_dialogue_turns / num_dialogues)
    statistics['AGENT_0']['objective_task_completion_percentage'] = \
        100 * float(ca.num_task_success / num_dialogues)

    print('\n\nDialogue Success Rate: {0}\nAverage Cumulative Reward: {1}'
          '\nAverage Turns: {2}'.
          format(statistics['AGENT_0']['dialogue_success_percentage'],
                 statistics['AGENT_0']['avg_cumulative_rewards'],
                 statistics['AGENT_0']['avg_turns']))

    return statistics

def arg_parse(args=None):
    """
    This function will parse the configuration file that was provided as a
    system argument into a dictionary.

    :return: a dictionary containing the parsed config file.
    """

    cfg_parser = None
    
    arg_vec = args if args else sys.argv

    # Parse arguments
    if len(arg_vec) < 3:
        print('WARNING: No configuration file.')
        config_yaml = 'config/train_reinforce.yaml'
        arg_vec+=['-config', config_yaml]

    test_mode = arg_vec[1] == '-t'

    if test_mode:
        return {'test_mode': test_mode}

    # Initialize random seed
    random.seed(time.time())

    cfg_filename = arg_vec[2]
    if isinstance(cfg_filename, str):
        if os.path.isfile(cfg_filename):
            # Choose config parser
            parts = cfg_filename.split('.')
            if len(parts) > 1:
                if parts[-1] == 'yaml':
                    with open(cfg_filename, 'r') as file:
                        cfg_parser = yaml.load(file, Loader=yaml.Loader)
                elif parts[1] == 'cfg':
                    cfg_parser = configparser.ConfigParser()
                    cfg_parser.read(cfg_filename)
                else:
                    raise ValueError('Unknown configuration file type: %s'
                                     % parts[1])
        else:
            raise FileNotFoundError('Configuration file %s not found'
                                    % cfg_filename)
    else:
        raise ValueError('Unacceptable value for configuration file name: %s '
                         % cfg_filename)

    tests = 1
    dialogues = 10
    interaction_mode = 'simulation'
    num_agents = 1

    if cfg_parser:
        dialogues = int(cfg_parser['DIALOGUE']['num_dialogues'])

        if 'interaction_mode' in cfg_parser['GENERAL']:
            interaction_mode = cfg_parser['GENERAL']['interaction_mode']

            if 'agents' in cfg_parser['GENERAL']:
                num_agents = int(cfg_parser['GENERAL']['agents'])

            elif interaction_mode == 'multi_agent':
                print('WARNING! Multi-Agent interaction mode selected but '
                      'number of agents is undefined in config.')

        if 'tests' in cfg_parser['GENERAL']:
            tests = int(cfg_parser['GENERAL']['tests'])

    return {'cfg_parser': cfg_parser,
            'tests': tests,
            'dialogues': dialogues,
            'interaction_mode': interaction_mode,
            'num_agents': num_agents,
            'test_mode': False}




if __name__ == '__main__':
    arguments = arg_parse()
    statistics = run_single_agent(
        arguments['cfg_parser'], arguments['dialogues'])

    pprint(f'Results:\n{statistics}')
