import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vss_task import VSS3v3SelfPlay
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import hydra

class Actor(nn.Module):
    def __init__(self, env, n_controlled_robots):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, n_controlled_robots * 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc_mu(x))
        return x


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


from rl_games import torch_runner
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
import gym
@hydra.main(config_name="VSSPPO", config_path="./cfg")
def main(cfg):
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
    player.restore()
    import pdb; pdb.set_trace()
    # task = VSS3v3SelfPlay(record=True)
    # runner = Runner(task)

    # obs = task.reset()
    # ep_count = 0
    # rw_sum = 0
    # len_sum = 0
    # frames = []
    # # while not task.gym.query_viewer_has_closed(task.viewer):
    # # while ep_count < 5000:
    # for i in range(1000):
    #     print(i)
    #     obs, rew, dones, info = task.step(runner.get_actions(obs))
    #     # task.render()
    #     frames.append(task.render())
    #     env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    #     if len(env_ids):
    #         ep_count += len(env_ids)
    #         rw_sum += rew[env_ids, 0].sum().item()
    #         len_sum += info['progress_buffer'][env_ids].sum().item()
    #         # print(ep_count)
    # clip = ImageSequenceClip(frames, fps=20)
    # clip.write_videofile('video.mp4')
    # print()
    # print(f'avg reward: {rw_sum / ep_count}')
    # print(f'avg length: {len_sum / ep_count}')


if __name__ == '__main__':
    main()
