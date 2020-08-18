
# Interspeech 2020 paper

## Comparison of Training Behavior and Performance of Reinforcement Learning based Policies for Dialogue Management
#### Stefan Hillmann, Tilo Himmelsbach, Benjamin Weiss

#### to reproduce the results
0. setup env: 
    0. `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    0. `bash Miniconda3-latest-Linux-x86_64.sh`
    1. `conda install -c anaconda pip`
    2. `pip install -r requirements.txt`
0. clone: `git clone git@gitlab.tubit.tu-berlin.de:OKS/plato.git`
0. checkout interspeech branch: `git checkout interspeech_2020`
0. copy domain-folder from alex-plato:`git clone https://gitlab.tubit.tu-berlin.de/OKS/alex-plato.git` and `cp -r alex-plato/experiments/essv_2020/domain plato/`
1. python [run_parallel_experiments.py](run_parallel_experiments.py)
2. (optional) [plot_results.py](plot_results.py)


### results
#### boxplots
![boxplots](results/5000_500/boxplot_%20True.png)

#### scatterplots
![boxplots](results/40000_4000/scatterplot_%20True.png)

### [analyze states](analyze_states.py)
```shell script
'num_states': {
            'q_learning': 5973,
            'q_learning_error_sim': 16461,
            'q_learning_error_sim_two_slots': 53652,
            'q_learning_two_slots': 21448}}
```