import argparse
from os import path, chdir, mkdir, getcwd
import glob
from subprocess import Popen, PIPE


def run_config(config_dir, plato_dir, mode=None):
    path_pattern = None
    if mode == 'train':
        path_pattern = config_dir + '/train_*.yaml'
    elif mode == 'eval':
        path_pattern = config_dir + '/eval_*.yaml'
    else:
        print('Mode {} is unknown.'.format(mode))
        return()

    train_configs = glob.glob(path_pattern)

    if len(train_configs) == 0:
        print('Found no training config files with path pattern "{}"!'.format(path_pattern))
    else:
        plato_script = plato_dir + '/runPlatoRDS.py'
        commands = [['python', plato_script, '-f', tc] for tc in train_configs]
        # processes = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in commands]
        processes = [Popen(cmd) for cmd in commands]

        print('Run configs for mode "{}"'.format(mode))
        status = [proc.wait() for proc in processes]
        print('Return status for mode "{}": {}'.format(mode, status))


def compress_logs(logs_dir):
    pkl_files = glob.glob(logs_dir + '/*.pkl')
    print('Files to be compressed: {}'.format(pkl_files))

    print('Compress PKL files.')
    commands = [['xz', '-k', file] for file in pkl_files]
    processes = [Popen(cmd) for cmd in commands]
    status = [p.wait() for p in processes]

    print('Return status of compressing processes: {}'.format(status))


def create_csv_files(logs_dir, plato_dir):
    print('Create CSV files from pickle logs.')

    pkl_files = glob.glob(logs_dir + '/*.pkl')
    print('Existing PKL files: {}'.format(pkl_files))

    # create csv file name from pkl_files
    csv_files = [pkl.replace('.pkl', '.csv') for pkl in pkl_files]
    print('CSV files to be generated: {}'.format(csv_files))

    file_tuples = list(zip(pkl_files, csv_files))

    script = plato_dir + '/log_to_csv.py'
    commands = [['python', script, '-i', ft[0], '-o', ft[1]] for ft in file_tuples]
    processes = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in commands]

    print('Generate CSV files')
    status = [proc.wait() for proc in processes]
    print('Status from CSV file generation: {}'.format(status))


if __name__ == '__main__':

    plato_directory = getcwd()

    parser = argparse.ArgumentParser('Run experiments with PLATO.')
    parser.add_argument('-b', required=True, type=str, help='Base folder for the experiment.')

    args = parser.parse_args()


    # check if input file exists
    base_dir = args.b
    if not path.exists(base_dir):
        print('Directory does not exist: {}'.format(base_dir))
        exit(-1)

    if path.isfile(base_dir):
        print('Base directory ({}) must be a directory!'.format(base_dir))
        exit(-1)

    # change working directory for the whole script
    chdir(base_dir)

    config_directory = 'configs'
    if not path.exists(config_directory) or not path.isdir(config_directory):
        print('Directory "{}" does not exist'.format(config_directory))
        exit(-1)

    log_directory = 'logs'
    if not path.exists(log_directory):
        print('Create directory "{}" in base directory ({})'.format(log_directory, base_dir))
        mkdir(log_directory)

    # create policies directory if it not exists
    pol_directory = 'policies'
    if not path.exists(pol_directory):
        print('Create directory "{}" in base directory ({})'.format(pol_directory, base_dir))
        mkdir(pol_directory)

    run_config(config_directory, plato_directory, mode='train')
    run_config(config_directory, plato_directory, mode='eval')
    compress_logs(log_directory)
    create_csv_files(log_directory, plato_directory)
