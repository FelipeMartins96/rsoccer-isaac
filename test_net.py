import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vss_task import VSS3v3SelfPlay
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


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


class Runner:
    def __init__(self, task):

        self.task = task
        self.net_1 = Actor(task, 1).to(task.rl_device)
        self.net_3 = Actor(task, 3).to(task.rl_device)
        self.net_3_alt = Actor(task, 3).to(task.rl_device)
        self.net_1.load_state_dict(torch.load('actor1controlled-stopped-onlygoal.pth'))
        self.net_3.load_state_dict(torch.load('actor3controlled-stopped-onlygoal.pth'))
        self.net_3_alt.load_state_dict(
            torch.load('actor3controlled-stopped-onlygoal-23-11-23-28.pth')
        )
        self.zero_action = torch.cat((task.zero_actions(), task.zero_actions()), dim=1)

    def get_actions(self, obs):
        with torch.no_grad():
            acts_b = self.net_3_alt(obs['obs'][:, 0, :])
            acts_y = self.net_1(obs['obs'][:, 1, :])
            # acts_y = self.net_1(obs['obs'][:, 0, :])

            action = self.zero_action.clone().reshape(
                self.task.num_envs, self.task.num_agents, -1
            )
            action[:, 0, 0:6] = acts_b
            # action[:, 1, 0:2] = acts_y
        return action.reshape(self.task.num_envs, -1).clone()


def main():
    task = VSS3v3SelfPlay()
    runner = Runner(task)

    obs = task.reset()
    ep_count = 0
    rw_sum = 0
    len_sum = 0
    frames = []
    # while not task.gym.query_viewer_has_closed(task.viewer):
    while ep_count < 5000:
        obs, rew, dones, info = task.step(runner.get_actions(obs))
        # frames.append(task.render())
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            ep_count += len(env_ids)
            rw_sum += rew[env_ids, 0].sum().item()
            len_sum += info['progress_buffer'][env_ids].sum().item()
            # print(ep_count)
    # clip = ImageSequenceClip(frames, fps=20)
    # clip.write_videofile('video.mp4')
    print()
    print(f'avg reward: {rw_sum / ep_count}')
    print(f'avg length: {len_sum / ep_count}')


if __name__ == '__main__':
    main()