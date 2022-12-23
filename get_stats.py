import pandas as pd

df = pd.read_csv('outputs/results.csv')

teams = ['ppo', 'ppo-x3', 'ppo-cma', 'ppo-dma', 'ou', 'zero']

goal_df = pd.DataFrame(index=teams, columns=teams)
goal_std_df = goal_df.copy()
steps_df = goal_df.copy()
steps_std_df = goal_df.copy()

for row in teams:
    for column in teams:
        # get mean and std of goal score and ep len for row team vs column team
        frame = df[df.team_a == row]
        frame = frame[frame.team_b == column]
        goal_df.loc[row][column] = frame.goal_score.mean()
        goal_std_df.loc[row][column] =frame.goal_score.std()
        steps_df.loc[row][column] = frame.episode_length.mean()
        steps_std_df.loc[row][column] = frame.episode_length.std()
goal_df.astype(float).round(5).to_csv('outputs/goal_stats.csv')
goal_std_df.astype(float).round(5).to_csv('outputs/goal_std_stats.csv')
steps_df.astype(float).round(5).to_csv('outputs/steps_stats.csv')
steps_std_df.astype(float).round(5).to_csv('outputs/steps_std_stats.csv')