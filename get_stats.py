import pandas as pd
import sys
import os

if len(sys.argv) != 2:
    print("need results path arg")

df = pd.read_csv(sys.argv[1])

algos = ['ppo', 'ppo-x3', 'ppo-cma', 'ppo-dma', 'ou', 'zero']
blue_exp = df.iloc[0].blue_experiment
yellow_exp = df.iloc[0].yellow_experiment

blue_algos = [f'{blue_exp}_' + a for a in algos]
yellow_algos = [f'{yellow_exp}_' + a for a in algos]

goal_df = pd.DataFrame(index=blue_algos, columns=yellow_algos)
goal_std_df = goal_df.copy()
steps_df = goal_df.copy()
steps_std_df = goal_df.copy()
score_df = goal_df.copy()

for row in blue_algos:
    for column in yellow_algos:
        # get mean and std of goal score and ep len for row team vs column team
        frame = df[df.blue_algo == row.split('_')[1]]
        frame = frame[frame.yellow_algo == column.split('_')[1]]
        goal_df.loc[row][column] = frame.goal_score.mean()
        goal_std_df.loc[row][column] = frame.goal_score.std()
        steps_df.loc[row][column] = frame.episode_length.mean()
        steps_std_df.loc[row][column] = frame.episode_length.std()
        score_df.loc[row][column] = frame.blue_score.mean()

results_path = os.path.join(os.path.dirname(sys.argv[1]), '0_summary.csv')
goal_df.astype(float).round(5).to_csv(results_path, mode='a')
goal_std_df.astype(float).round(5).to_csv(results_path, mode='a')
steps_df.astype(float).round(5).to_csv(results_path, mode='a')
steps_std_df.astype(float).round(5).to_csv(results_path, mode='a')
score_df.astype(float).round(5).to_csv(results_path, mode='a')
scores_df = scores_df.mean(1).astype(float).round(5).to_csv(results_path, mode='a')
