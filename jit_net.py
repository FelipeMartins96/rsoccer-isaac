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


task = VSS3v3()

actor = Actor(task)
actor.load_state_dict(torch.load('actor-70.pt'))
obs = task.reset()['obs'][0].cpu()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(actor, obs)

traced_script_module.save("traced_actor-70.pt")
