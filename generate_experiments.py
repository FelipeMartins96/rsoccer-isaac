import os
from collections import namedtuple
from itertools import combinations_with_replacement

TOTAL_EPS = 1000
## Get folders list
runs = os.listdir('runs')

team = namedtuple('team', ['name', 'seed', 'tag', 'checkpoint'])

## Get nn path, last and best
teams = []
for run in runs:
    run_path = os.path.join('runs', run, 'nn')

    ## get best nn
    name, seed = run.split('_')
    path = os.path.abspath(os.path.join(run_path, f'{run}.pth'))
    teams.append(team(name, seed, 'best', path))

    ## get last nn
    for ckpt in [p for p in os.listdir(run_path) if f'ep_{TOTAL_EPS}_rew' in p]:
        path = os.path.abspath(os.path.join(run_path, ckpt))
        teams.append(team(name, seed, 'last', path))


## Add ppo-x3 cases
vss3_teams = []
for t in teams:
    if t.name == 'ppo':
        vss3_teams.append(team(f'ppo-x3', t.seed, t.tag, t.checkpoint))
teams += vss3_teams

## Add OU and Zero cases
teams += [team('ou', '', '', ''), team('zero', '', '', '')]

## Get combinations
teams_combinations = combinations_with_replacement(teams, 2)

f = open('test_nets.sh', 'w')
## For each pair
i = 0
for team_comb in teams_combinations:
    team_a, team_b = team_comb
    f.write(
        f'python test_net.py index={i} blue_team={team_a.name} blue_ckpt={team_a.checkpoint}  blue_seed={team_a.seed} blue_tag={team_a.tag} yellow_team={team_b.name} yellow_ckpt={team_b.checkpoint} yellow_seed={team_b.seed} yellow_tag={team_b.tag}\n'
    )
    i += 1
