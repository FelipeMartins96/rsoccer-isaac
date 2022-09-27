from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymapi


def _parse_cfg(cfg):
    cfg['env'] = {}
    cfg['env']['numEnvs'] = cfg['n_envs']
    cfg['env']['numObservations'] = 0
    cfg['env']['numActions'] = 0

    cfg['sim'] = {}
    cfg['sim']['use_gpu_pipeline'] = cfg['use_gpu_pipeline']
    cfg['sim']['up_axis'] = 'z'
    cfg['sim']['dt'] = 1 / 50
    cfg['sim']['gravity'] = [0, 0, -9.81]

    return cfg


class VSS3v3(VecTask):
    def __init__(self, cfg):
        self.cfg = _parse_cfg(cfg)
        self.max_episode_length = 500

        super().__init__(
            config=self.cfg,
            rl_device=self.cfg['rl_device'],
            sim_device=self.cfg['sim_device'],
            graphics_device_id=self.cfg['graphics_device_id'],
            headless=self.cfg['headless'],
            virtual_screen_capture=self.cfg['virtual_screen_capture'],
            force_render=self.cfg['force_render'],
        )

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass
