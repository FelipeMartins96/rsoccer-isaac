import os
from collections import namedtuple
from itertools import combinations_with_replacement

team = namedtuple('team', ['experiment', 'seed', 'algo', 'checkpoint'])

EXPERIMENT_A = 'teste'
EXPERIMENT_B = 'teste-2'

## Get nn path, last and best
teams_a = []
teams_b = []

## Get folders list
runs = os.listdir('runs')
for run in [r for r in runs if EXPERIMENT_A in r]:
    run_path = os.path.join('runs', run, 'nn')

    # experiment-name_algo_seed
    ## get best nn
    experiment = run.split('_')[0]
    algo = run.split('_')[1]
    seed = run.split('_')[-1]
    path = os.path.abspath(os.path.join(run_path, f'{run}.pth'))
    teams_a.append(team(experiment, seed, algo, path))

    if algo == 'ppo':
        teams_b.append(team(experiment, seed, algo + '-x3', path))

for run in [r for r in runs if EXPERIMENT_B in r]:
    run_path = os.path.join('runs', run, 'nn')

    # experiment-name_algo_seed
    ## get best nn
    experiment = run.split('_')[0]
    algo = run.split('_')[1]
    seed = run.split('_')[-1]
    path = os.path.abspath(os.path.join(run_path, f'{run}.pth'))
    teams_b.append(team(experiment, seed, algo, path))

    if algo == 'ppo':
        teams_b.append(team(experiment, seed, algo + '-x3', path))


## Add OU and Zero cases
teams_a += [team(EXPERIMENT_A, 0, 'ou', ''), team(EXPERIMENT_A, 0, 'zero', '')]
teams_b += [team(EXPERIMENT_B, 0, 'ou', ''), team(EXPERIMENT_B, 0, 'zero', '')]

f = open('test_nets.sh', 'w')

## For each pair
i = 0
for t_a in teams_a:
    for t_b in teams_b:
        f.write(
            f'python test_net.py index={i} blue_exp={t_a.experiment} blue_algo={t_a.algo} blue_seed={t_a.seed} blue_ckpt={t_a.checkpoint} yellow_exp={t_b.experiment} yellow_algo={t_b.algo} yellow_seed={t_b.seed} yellow_ckpt={t_b.checkpoint}\n'
        )
        i+=1
