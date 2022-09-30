import os
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymapi
import numpy as np


def get_cfg():
    cfg = {
        'n_envs': 1,
        'rl_device': 'cuda:0',
        'sim_device': 'cuda:0',
        'use_gpu_pipeline': False,
        'graphics_device_id': 0,
        'headless': False,
        'virtual_screen_capture': False,
        'force_render': True,
        'physics_engine': 'physx',
    }

    cfg['env'] = {
        'numEnvs': cfg['n_envs'],
        'numObservations': 0,
    }

    cfg['sim'] = {
        'use_gpu_pipeline': cfg['use_gpu_pipeline'],
        'up_axis': 'z',
        'dt': 1 / 50,
        'gravity': [0, 0, -9.81],
    }

    return cfg


class VSS3v3(VecTask):
    def __init__(self):
        self.cfg = get_cfg()
        self.max_episode_length = 500

        self.n_blue_robots = 1
        self.n_yellow_robots = 0
        self.env_total_width = 2
        self.env_total_height = 1.5

        self.cfg['env']['numActions'] = 2 * (self.n_blue_robots + self.n_yellow_robots)

        super().__init__(
            config=self.cfg,
            rl_device=self.cfg['rl_device'],
            sim_device=self.cfg['sim_device'],
            graphics_device_id=self.cfg['graphics_device_id'],
            headless=self.cfg['headless'],
            virtual_screen_capture=self.cfg['virtual_screen_capture'],
            force_render=self.cfg['force_render'],
        )

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.0, -0.2, 4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._add_ground()
        env_bounds_low = (
            -gymapi.Vec3(self.env_total_width, self.env_total_height, 0.0) / 2
        )
        env_bounds_high = (
            gymapi.Vec3(self.env_total_width, self.env_total_height, 2.0) / 2
        )
        for i in range(self.num_envs):
            _env = self.gym.create_env(
                self.sim, env_bounds_low, env_bounds_high, int(np.sqrt(self.num_envs))
            )

            self._add_ball(_env, i)
            for j in range(self.n_blue_robots):
                self._add_robot(_env, i, gymapi.Vec3(0.0, 0.0, 0.3))
            for i in range(self.n_yellow_robots):
                self._add_robot(_env, i, gymapi.Vec3(0.3, 0.3, 0.0))
            self._add_field(_env, i)

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def _add_ground(self):
        pp = gymapi.PlaneParams()
        pp.distance = 0.0
        pp.dynamic_friction = 1.0
        pp.normal = gymapi.Vec3(
            0, 0, 1
        )  # defaultgymapi.Vec3(0.000000, 1.000000, 0.000000)
        pp.restitution = 0.0
        pp.segmentation_id = 0
        pp.static_friction = 1.0
        self.gym.add_ground(self.sim, pp)

    def _add_ball(self, env, env_id):
        options = gymapi.AssetOptions()
        options.density = 1130.0  # 0.046 kg
        color = gymapi.Vec3(1.0, 0.4, 0.0)
        radius = 0.02134
        pose = gymapi.Transform(p=gymapi.Vec3(0.3, 0.1, radius))
        asset = self.gym.create_sphere(self.sim, radius, options)
        ball = self.gym.create_actor(
            env=env, asset=asset, pose=pose, group=env_id, filter=0b01
        )
        self.gym.set_rigid_body_color(env, ball, 0, gymapi.MESH_VISUAL, color)

    def _add_robot(self, env, env_id, color):
        options = gymapi.AssetOptions()
        root = os.path.dirname(os.path.abspath(__file__))
        rbt_asset = self.gym.load_asset(
            sim=self.sim,
            rootpath=root,
            filename='assets/vss_robot.urdf',
            options=options,
        )
        body, left_wheel, right_wheel = 0, 1, 2
        initial_height = 0.028  # _z dimension
        pose = gymapi.Transform(p=gymapi.Vec3(0, 0.0, initial_height))
        robot = self.gym.create_actor(
            env=env, asset=rbt_asset, pose=pose, group=env_id, filter=0b00, name='robot'
        )
        self.gym.set_rigid_body_color(env, robot, body, gymapi.MESH_VISUAL, color)
        props = self.gym.get_actor_rigid_shape_properties(env, robot)
        props[body].friction = 0.0
        props[body].filter = 0b0
        props[left_wheel].filter = 0b11
        props[right_wheel].filter = 0b11
        self.gym.set_actor_rigid_shape_properties(env, robot, props)

        props = self.gym.get_actor_dof_properties(env, robot)
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.0)
        props["damping"].fill(200.0)
        self.gym.set_actor_dof_properties(env, robot, props)

    def _add_field(self, env, env_id):
        # Using procedural assets because with an urdf file rigid contacts were not being drawn
        # _width (x), _width (_y), Depth (_z)
        total_width = self.env_total_width
        total_height = self.env_total_height
        field_width = 1.5
        field_height = 1.3
        goal_width = 0.1
        goal_height = 0.4
        walls_depth = 0.1  # on rules its 0.05

        options = gymapi.AssetOptions()
        options.fix_base_link = True
        color = gymapi.Vec3(0.2, 0.2, 0.2)

        # Side Walls (sw)
        def add_side_walls():
            sw_width = total_width
            sw_height = (total_height - field_height) / 2
            sw_x = 0
            sw_y = (field_height + sw_height) / 2
            sw_z = walls_depth / 2
            swDirs = [(1, 1), (1, -1)]  # Top and Bottom

            sw_asset = self.gym.create_box(
                self.sim, sw_width, sw_height, walls_depth, options
            )

            for dir_x, dir_y in swDirs:
                swPose = gymapi.Transform(
                    p=gymapi.Vec3(dir_x * sw_x, dir_y * sw_y, sw_z)
                )
                swActor = self.gym.create_actor(
                    env=env, asset=sw_asset, pose=swPose, group=env_id, filter=0b10
                )
                self.gym.set_rigid_body_color(
                    env, swActor, 0, gymapi.MESH_VISUAL, color
                )

        # End Walls (ew)
        def add_end_walls():
            ew_width = (total_width - field_width) / 2
            ew_height = (field_height - goal_height) / 2
            ew_x = (field_width + ew_width) / 2
            ew_y = (field_height - ew_height) / 2
            ew_z = walls_depth / 2
            ewDirs = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # Corners

            ew_asset = self.gym.create_box(
                self.sim, ew_width, ew_height, walls_depth, options
            )

            for dir_x, dir_y in ewDirs:
                ewPose = gymapi.Transform(
                    p=gymapi.Vec3(dir_x * ew_x, dir_y * ew_y, ew_z)
                )
                ewActor = self.gym.create_actor(
                    env=env, asset=ew_asset, pose=ewPose, group=env_id, filter=0b10
                )
                self.gym.set_rigid_body_color(
                    env, ewActor, 0, gymapi.MESH_VISUAL, color
                )

        # Goal Walls (gw)
        def add_goal_walls():
            gw_width = ((total_width - field_width) / 2) - goal_width
            gw_height = goal_height
            gw_x = (total_width - gw_width) / 2
            gw_y = 0
            gw_z = walls_depth / 2
            gwDirs = [(-1, 1), (1, 1)]  # left and right

            gw_asset = self.gym.create_box(
                self.sim, gw_width, gw_height, walls_depth, options
            )

            for dir_x, dir_y in gwDirs:
                gwPose = gymapi.Transform(
                    p=gymapi.Vec3(dir_x * gw_x, dir_y * gw_y, gw_z)
                )
                gwActor = self.gym.create_actor(
                    env=env, asset=gw_asset, pose=gwPose, group=env_id, filter=0b10
                )
                self.gym.set_rigid_body_color(
                    env, gwActor, 0, gymapi.MESH_VISUAL, color
                )

        add_side_walls()
        add_end_walls()
        add_goal_walls()
