import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vss_task import VSS3v3SelfPlay
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import hydra
from rl_games import torch_runner
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
import gym
from functools import partial

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


class Runner:
    def __init__(self, task):

        self.task = task
        self.net_1 = Actor(task, 1).to(task.rl_device)
        # self.net_3 = Actor(task, 3).to(task.rl_device)
        # self.net_3_alt = Actor(task, 3).to(task.rl_device)
        self.net_1.load_state_dict(
            torch.load(
                '/home/fbm2/isaac/rsoccer-isaac/runs/Nov08_14-55-14_SARCO-021-vs-ou/actor1-vs-ou.pth'
            )
        )
        # self.net_3.load_state_dict(torch.load('actor3controlled-stopped-onlygoal.pth'))
        # self.net_3_alt.load_state_dict(
        # torch.load('actor3controlled-stopped-onlygoal-23-11-23-28.pth')
        # )
        self.zero_action = task.zero_actions()
        self.noise = task.zero_actions()

    def get_actions(self, obs):
        with torch.no_grad():
            # acts_b = self.net_3_alt(obs['obs'][:, 0, :])
            acts_b = self.net_1(obs['obs'][:, 0, :])
            # acts_y = self.net_1(obs['obs'][:, 0, :])

            self.noise = random_ou(self.noise)
            # action = self.zero_action.clone()
            action = self.noise.clone()
            # .reshape(
            #     self.task.num_envs, self.task.num_agents, -1
            # )
            action[:, 0:2] = acts_b
            # action[:, 1, 0:2] = acts_y
        return action



def get_vss_player(cfg, ckpt):
    cfg = omegaconf_to_dict(cfg)
    cfg['params']['config']['env_info'] = {
        'observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32),
        'action_space': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        'agents': 1,
        'value_size': 1
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
        'observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32),
        'action_space': gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        'agents': 1,
        'value_size': 1
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
        'observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32),
        'action_space': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        'agents': 1,
        'value_size': 1
    }
    runner = torch_runner.Runner()
    runner.load(cfg)
    print_dict(runner.params)
    player = runner.create_player()
    player.restore(ckpt)
    return player

# get actions

def get_obs_with_perms(obs):
    _obs = obs.repeat_interleave(3, dim=0)
    permutations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    permutations = torch.tensor(permutations, device=obs.device)
    _obs[:, 4:25] = obs[:, 4:25].view(-1, 3, 7)[:, permutations].view(-1, 21)
    _obs[:, -6:] = obs[:, -6:].view(-1, 3, 2)[:, permutations].view(-1, 6)
    return _obs

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

@hydra.main(config_name="config", config_path="./cfg")
def main(cfg):
    task = VSS3v3SelfPlay(record=False, num_envs=cfg.num_envs, has_grad=False)
    
    p_vss = get_vss_player(cfg.vss, '/home/fbm2/isaac/rsoccer-isaac/ppo-0.pth')
    p_cma = get_vsscma_player(cfg.vss, '/home/fbm2/isaac/rsoccer-isaac/ppo-CMA-1.pth')
    p_dma = get_vssdma_player(cfg.vssdma, '/home/fbm2/isaac/rsoccer-isaac/ppo-DMA.pth')

    get_vss1_actions = partial(_get_vss1_actions, player=p_vss)
    get_vss3_actions = partial(_get_dma_actions, player=p_vss)
    get_cma_actions = partial(_get_cma_actions, player=p_cma)
    get_dma_actions = partial(_get_dma_actions, player=p_dma)

    # TODO: Generate gif for each case and run statistics
    
    get_blue_actions = get_cma_actions
    get_yellow_actions = get_dma_actions

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
        # Get actions yellow
        yellow_actions = actions.view(-1, 2, 6)[:, 1, :]
        yellow_actions = get_yellow_actions(obs['obs'][:, 1, :], yellow_actions)

        obs, rew, dones, info = task.step(actions)
        
        frames.append(task.render()) if cfg.record else None
        
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            ep_count += len(env_ids)
            rw_sum += rew[env_ids, 0].sum().item()
            len_sum += info['progress_buffer'][env_ids].sum().item()
            print(ep_count) if ep_count % 25 == 0 else None
    
    if cfg.record:
        clip = ImageSequenceClip(frames, fps=20)
        clip.write_videofile(f'{cfg.experiment}.mp4')
   
    print()
    print(cfg.experiment)
    print(f'avg reward: {rw_sum / ep_count}')
    print(f'avg length: {len_sum / ep_count}')


if __name__ == '__main__':
    main()
