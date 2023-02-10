import isaacgym
import torch
import numpy as np
from vss_task import VSS3v3SelfPlay
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import hydra
from rl_games import torch_runner
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
import gym
from functools import partial
import os
import pandas as pd


def random_ou(prev):
    ou_theta = 0.1
    ou_sigma = 0.15
    noise = (
        prev
        - ou_theta * prev
        + torch.normal(
            0.0,
            ou_sigma,
            size=prev.size(),
            device=prev.device,
            requires_grad=False,
        )
    )
    return noise.clamp(-1.0, 1.0)


def get_vss_player(cfg, ckpt):
    cfg = omegaconf_to_dict(cfg)
    cfg['params']['config']['env_info'] = {
        'observation_space': gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        ),
        'action_space': gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ),
        'agents': 1,
        'value_size': 1,
    }
    runner = torch_runner.Runner()
    runner.load(cfg)
    print_dict(runner.params)
    player = runner.create_player()
    player.restore(ckpt)
    return player


def get_vsscma_player(cfg, ckpt):
    cfg = omegaconf_to_dict(cfg)
    cfg['params']['config']['env_info'] = {
        'observation_space': gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        ),
        'action_space': gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        ),
        'agents': 1,
        'value_size': 1,
    }
    runner = torch_runner.Runner()
    runner.load(cfg)
    print_dict(runner.params)
    player = runner.create_player()
    player.restore(ckpt)
    return player


def get_vssdma_player(cfg, ckpt):
    cfg = omegaconf_to_dict(cfg)
    cfg['params']['config']['env_info'] = {
        'observation_space': gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        ),
        'action_space': gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ),
        'agents': 1,
        'value_size': 1,
    }
    runner = torch_runner.Runner()
    runner.load(cfg)
    print_dict(runner.params)
    player = runner.create_player()
    player.restore(ckpt)
    return player



def get_obs_with_perms(obs):
    _obs = obs.repeat_interleave(3, dim=0)
    permutations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    permutations = torch.tensor(permutations, device=obs.device)
    _obs[:, 4:25] = obs[:, 4:25].view(-1, 3, 7)[:, permutations].view(-1, 21)
    _obs[:, -6:] = obs[:, -6:].view(-1, 3, 2)[:, permutations].view(-1, 6)
    return _obs

# get actions

def _get_vss1_actions(obs, actions, player):
    player.has_batch_dimension = True
    _action = player.get_action(obs, is_determenistic=True)
    actions[:, :2] = _action
    return actions


def _get_dma_actions(obs, actions, player):
    player.has_batch_dimension = True
    _obs = get_obs_with_perms(obs)
    _action = player.get_action(_obs, is_determenistic=True)
    actions[:, :6] = _action.view(-1, 6)
    return actions


def _get_cma_actions(obs, actions, player):
    player.has_batch_dimension = True
    _actions = player.get_action(obs, is_determenistic=True)
    actions[:, :6] = _actions
    return actions


def _get_ou_actions(obs, actions, player):
    return actions


def _get_zero_actions(obs, actions, player):
    actions = actions * 0
    return actions


def get_team_actions(cfg, algo, checkpoint):
    if algo == 'ou':
        return partial(_get_ou_actions, player=None)
    elif algo == 'zero':
        return partial(_get_zero_actions, player=None)
    elif algo == 'ppo':
        return partial(_get_vss1_actions, player=get_vss_player(cfg.vss, checkpoint))
    elif algo == 'ppo-x3':
        return partial(_get_dma_actions, player=get_vss_player(cfg.vss, checkpoint))
    elif algo == 'ppo-dma':
        return partial(
            _get_dma_actions, player=get_vssdma_player(cfg.vssdma, checkpoint)
        )
    elif algo == 'ppo-cma':
        return partial(_get_cma_actions, player=get_vsscma_player(cfg.vss, checkpoint))
    else:
        raise ValueError(f'Unknown algo: {algo}')


@hydra.main(config_name="config", config_path="./cfg")
def main(cfg):
    experiment = f'{cfg.blue_algo}_{cfg.blue_seed}x{cfg.yellow_algo}_{cfg.yellow_seed}'

    task = VSS3v3SelfPlay(record=cfg.record, num_envs=cfg.num_envs, has_grad=False)

    get_blue_actions = get_team_actions(cfg, cfg.blue_algo, cfg.blue_ckpt)
    get_yellow_actions = get_team_actions(cfg, cfg.yellow_algo, cfg.yellow_ckpt)

    obs = task.reset()
    ep_count = 0
    rw_sum = 0
    len_sum = 0
    frames = []
    actions = task.zero_actions()

    while ep_count < cfg.num_eps:

        actions = random_ou(actions)
        
        # Get actions blue
        blue_actions = actions.view(-1, 2, 6)[:, 0, :]
        blue_actions = get_blue_actions(obs['obs'][:, 0, :], blue_actions)
        actions.view(-1, 2, 6)[:, 0, :] = blue_actions
        # Get actions yellow
        yellow_actions = actions.view(-1, 2, 6)[:, 1, :]
        yellow_actions = get_yellow_actions(obs['obs'][:, 1, :], yellow_actions)
        actions.view(-1, 2, 6)[:, 1, :] = yellow_actions

        obs, rew, dones, info = task.step(actions)

        frames.append(task.render()) if cfg.record and len(frames) < 500 else None

        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            ep_count += len(env_ids)
            rw_sum += rew[env_ids, 0].sum().item()
            len_sum += info['progress_buffer'][env_ids].sum().item()

    output_path = os.path.join('outputs', f'{cfg.blue_exp}_{cfg.yellow_exp}')
    if cfg.record:
        os.makedirs(output_path, exist_ok=True)
        clip = ImageSequenceClip(frames, fps=20)
        clip.write_videofile(os.path.join(output_path, f'{experiment}.mp4'))

    result = {
        'index': cfg.index,
        'blue_experiment': cfg.blue_exp,
        'blue_algo': cfg.blue_algo,
        'blue_seed': cfg.blue_seed,
        'yellow_experiment': cfg.yellow_exp,
        'yellow_algo': cfg.yellow_algo,
        'yellow_seed': cfg.yellow_seed,
        'goal_score': rw_sum / ep_count,
        'episode_length': len_sum / ep_count,
    }
    df = pd.DataFrame(result, index=[0])

    df.to_csv(
        os.path.join(output_path, 'results.csv'),
        mode='a',
        index=False,
        header=not os.path.exists(os.path.join(output_path, 'results.csv')),
    )


if __name__ == '__main__':
    main()
