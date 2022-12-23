import os
from collections import namedtuple
from itertools import combinations_with_replacement

TOTAL_EPS = 1000
## Get folders list
runs = os.listdir('runs')

team = namedtuple('team', ['name', 'seed', 'algo', 'checkpoint'])

## Get nn path, last and best
teams = []
for run in runs:
    run_path = os.path.join('runs', run, 'nn')

    ## get best nn
    algo = run.split('_')[0]
    name = run.split('_')[:-1].join('_')
    seed = run.split('_')[-1]
    path = os.path.abspath(os.path.join(run_path, f'{run}.pth'))
    teams.append(team(name, seed, algo, path))

    # ## get last nn
    # for ckpt in [p for p in os.listdir(run_path) if f'ep_{TOTAL_EPS}_rew' in p]:
    #     path = os.path.abspath(os.path.join(run_path, ckpt))
    #     teams.append(team(name, seed, 'last', path))


path = os.path.abspath('ppo-0.pth')
teams.append(team('ppo_oldEnv', '0', 'ppo', path))
path = os.path.abspath('ppo-CMA-1.pth')
teams.append(team('ppo-cma_oldEnv', '1', 'ppo-cma', path))
path = os.path.abspath('ppo-DMA.pth')
teams.append(team('ppo-dma_oldEnv', '', 'ppo-dma', path))

## Add ppo-x3 cases
vss3_teams = []
for t in teams:
    if t.algo == 'ppo':
        vss3_teams.append(team(t.name+'_x3', t.seed, t.algo+'-x3', t.checkpoint))
teams += vss3_teams

## Add OU and Zero cases
teams += [team('ou', '', 'ou', ''), team('zero', '', 'zero', '')]

## Get combinations
teams_combinations = combinations_with_replacement(teams, 2)

f = open('test_nets.sh', 'w')
## For each pair
i = 0
for team_comb in teams_combinations:
    team_a, team_b = team_comb

    if 'oldEnv' in team_a.name and 'oldEnv' in team_b.name:
        continue
    if not 'oldEnv' in team_a.name and not 'oldEnv' in team_b.name:
        continue
    f.write(
        f'python test_net.py index={i} blue_team={team_a.name} blue_ckpt={team_a.checkpoint}  blue_seed={team_a.seed} blue_algo={team_a.algo} yellow_team={team_b.name} yellow_ckpt={team_b.checkpoint} yellow_seed={team_b.seed} yellow_algo={team_b.algo}\n'
    )
    i += 1
