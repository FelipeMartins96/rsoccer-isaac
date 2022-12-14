import isaacgym
import torch
from vss_task import VSS3v3SelfPlay


def play_random_policy():
    task = VSS3v3SelfPlay()

    def random_vec_actions():
        a = torch.rand(
            (task.num_envs) + task.action_space.shape,
            device=task.rl_device,
        )
        return (a - 0.5) * 2

    def constant_value(value):
        a = torch.ones(
            (task.num_envs,) + task.action_space.shape, device=task.rl_device
        )
        return a * value

    while not task.gym.query_viewer_has_closed(task.viewer):
        task.step(task.zero_actions())
        task.render()


if __name__ == '__main__':
    play_random_policy()
