import hydra
import isaacgym
import torch
from vss_task import VSS3v3
from isaacgymenvs.utils.reformat import omegaconf_to_dict


@hydra.main(config_path=".", config_name="task")
def play_random_policy(cfg_task):
    task = VSS3v3(omegaconf_to_dict(cfg_task))

    def random_vec_actions():
        a = torch.rand(
            (cfg_task.n_envs,) + task.action_space.shape, device=cfg_task.rl_device
        )
        return (a - 0.5) * 2

    while not task.gym.query_viewer_has_closed(task.viewer):
        task.step(random_vec_actions())


if __name__ == '__main__':
    play_random_policy()
