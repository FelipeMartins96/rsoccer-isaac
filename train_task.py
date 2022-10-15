import hydra
import gym
import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.learning.replay_buffer import ReplayBuffer
from vss_task import VSS3v3

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc_mu(x))
        return x


def train() -> None:
    task = VSS3v3()

    writer = SummaryWriter()

    device = task.cfg['rl_device']
    lr = 3e-4
    total_timesteps = 1000000
    learning_starts = 25e3
    batch_size = 2048
    gamma = 0.99
    tau = 0.005

    actor = Actor(task).to(device=device)
    qf1 = QNetwork(task).to(device=device)
    qf1_target = QNetwork(task).to(device=device)
    target_actor = Actor(task).to(device=device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=lr)

    rb = ReplayBuffer(2000000, device)
    start_time = time.time()
    episodic_return = torch.zeros(task.num_envs, device=device)
    episodic_length = torch.zeros(task.num_envs, device=device)
    # TRY NOT TO MODIFY: start the game
    obs = deepcopy(task.reset())
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        if rb.get_total_count() < learning_starts:
            actions = torch.tensor(
                [task.action_space.sample() for _ in range(task.num_envs)],
                dtype=torch.float32,
                device=device,
            )
        else:
            with torch.no_grad():
                actions = actor(obs['obs'])
                actions += torch.normal(
                    torch.zeros_like(actions, dtype=torch.float32, device=device),
                    1.0 * 0.4,
                )

        actions = torch.clamp(actions, -1.0, 1.0)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = task.step(actions)

        writer.add_scalar(
            "charts/mean_action_m1", actions[:, 0].mean().item(), global_step
        )
        writer.add_scalar(
            "charts/mean_action_m2", actions[:, 1].mean().item(), global_step
        )

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        real_next_obs = next_obs['obs'].clone()
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            writer.add_scalar(
                "charts/episodic_length",
                infos['progress_buffer'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_goal",
                infos['terminal_rewards']['goal'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_grad",
                infos['terminal_rewards']['grad'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_energy",
                infos['terminal_rewards']['energy'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_move",
                infos['terminal_rewards']['move'][env_ids].mean(),
                global_step,
            )
            real_next_obs[env_ids] = infos["terminal_observation"][env_ids]
            dones = dones.logical_and(infos["time_outs"].logical_not())

        # TRY NOT TO MODIFY: save data to replay buffer;
        rb.store(
            {
                'observations': obs['obs'],
                'next_observations': real_next_obs,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
            }
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = deepcopy(next_obs)

        # ALGO LOGIC: training.
        if rb.get_total_count() > learning_starts:
            data = rb.sample(batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data['next_observations'])
                qf1_next_target = qf1_target(
                    data['next_observations'], next_state_actions
                )
                next_q_value = data['rewards'].flatten() + (
                    1 - data['dones'].flatten()
                ) * gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data['observations'], data['actions']).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            actor_loss = -qf1(data['observations'], actor(data['observations'])).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(
                actor.parameters(), target_actor.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            if global_step % 10000 == 0:
                torch.save(
                    actor.state_dict(),
                    f"actor.pt",
                )


if __name__ == "__main__":
    train()
