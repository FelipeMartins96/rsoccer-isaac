import os
from typing import Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from torch import Tensor
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis


def get_cfg():
    cfg = {
        'n_envs': 2,
        'rl_device': 'cpu',
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
    }

    cfg['sim'] = {
        'use_gpu_pipeline': cfg['use_gpu_pipeline'],
        'up_axis': 'z',
        'dt': 1 / 50,
        'gravity': [0, 0, -9.81],
    }

    cfg['sim']['physx'] = {
        'use_gpu': False,
    }

    return cfg


class VSS3v3(VecTask):
    def __init__(self):
        self.cfg = get_cfg()
        self.max_episode_length = 30

        self.n_blue_robots = 1
        self.n_yellow_robots = 0
        self.env_total_width = 2
        self.env_total_height = 1.5
        self.robot_max_wheel_rad_s = 68
        self.field_width = 1.5
        self.field_height = 1.3
        self.goal_height = 0.4

        self.cfg['env']['numActions'] = 2 * (self.n_blue_robots + self.n_yellow_robots)
        self.cfg['env']['numObservations'] = (
            4 + (self.n_blue_robots + self.n_yellow_robots) * 13
        )

        super().__init__(
            config=self.cfg,
            rl_device=self.cfg['rl_device'],
            sim_device=self.cfg['sim_device'],
            graphics_device_id=self.cfg['graphics_device_id'],
            headless=self.cfg['headless'],
            virtual_screen_capture=self.cfg['virtual_screen_capture'],
            force_render=self.cfg['force_render'],
        )

        self._acquire_tensors()
        self._refresh_tensors()

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

    def pre_physics_step(self, _actions):
        # reset progress_buf for envs reseted on previous step
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.progress_buf[env_ids] = 0

        self.actions = _actions.to(self.device)
        act = self.actions * self.robot_max_wheel_rad_s
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(act))

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_rewards_and_dones()

        # Save observations previously to resets
        self.compute_observations()
        self.extras["terminal_observation"] = self.obs_buf.clone().to(self.rl_device)

        self.reset_dones()
        self.compute_observations()

    def compute_rewards_and_dones(self):
        pre_rew, self.reset_buf[:] = compute_vss_reward_and_dones(
            self.ball_pos,
            self.robot_pos,
            self.actions * 0,
            self.reset_buf,
            self.progress_buf,
            self.yellow_goal,
            self.max_episode_length,
            self.field_width,
            self.goal_height,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self._refresh_tensors()
        self.rew_buf[:], self.reset_buf[:] = compute_vss_reward_and_dones(
            self.ball_pos,
            self.robot_pos,
            self.actions * 0,
            self.reset_buf,
            self.progress_buf,
            self.yellow_goal,
            self.max_episode_length,
            self.field_width,
            self.goal_height,
            1.0,
            1.0,
            1.0,
            1.0,
        )

        self.rew_buf = self.rew_buf - pre_rew

    def compute_observations(self):
        self.obs_buf[..., :2] = self.ball_pos
        self.obs_buf[..., 2:4] = self.ball_vel
        self.obs_buf[..., 4:] = self.robot_state

    def reset_dones(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            # Reset env state
            self.root_state[env_ids] = self.env_reset_root_state

            # randomize positions
            rand_pos = (
                torch.rand((len(env_ids), 2, 2), dtype=torch.float, device=self.device)
                - 0.5
            ) * self.field_scale
            self.ball_pos[env_ids] = rand_pos[:, 0]
            self.robot_pos[env_ids] = rand_pos[:, 1]

            # randomize rotations
            rand_angles = torch_rand_float(
                -np.pi, np.pi, (len(env_ids), 1), device=self.device
            )
            self.robot_rotation[env_ids] = quat_from_angle_axis(
                rand_angles[:, 0], self.z_axis
            )

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_state)
            )

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
        pose = gymapi.Transform(p=gymapi.Vec3(0.2, 0.0, radius))
        asset = self.gym.create_sphere(self.sim, radius, options)
        ball = self.gym.create_actor(
            env=env, asset=asset, pose=pose, group=env_id, filter=0b01, name='ball'
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
        field_width = self.field_width
        field_height = 1.3
        goal_width = 0.1
        goal_height = self.goal_height
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

    def _acquire_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.field_scale = torch.tensor(
            [self.field_width, self.field_height],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # TODO: working only for one robot

        n_balls = 1
        n_robots = self.n_blue_robots + self.n_yellow_robots
        n_field_actors = 8  # 2 side walls, 4 end walls, 2 goal walls
        self.num_actors = n_balls + n_robots + n_field_actors
        self.ball = 0
        self.robot = 1
        # self.blue_robots = slice(n_balls, n_balls + self.n_blue_robots)
        # self.yellow_robots = slice(n_balls + self.n_blue_robots, n_balls + n_robots)

        # shape = (num_envs * num_actors, 13)
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_state).view(
            self.num_envs, self.num_actors, 13
        )
        self.robot_state = self.root_state[:, self.robot, :]

        self.root_pos = self.root_state[..., 0:2]
        self.robot_pos = self.root_pos[:, self.robot, :]
        self.ball_pos = self.root_pos[:, self.ball, :]

        self.root_rotation = self.root_state[..., 3:7]
        self.robot_rotation = self.root_rotation[:, self.robot, :]

        self.root_vel = self.root_state[..., 7:9]
        self.robot_vel = self.root_vel[:, self.robot, :]
        self.ball_vel = self.root_vel[:, self.ball, :]

        self._refresh_tensors()
        self.env_reset_root_state = self.root_state.clone()
        self.z_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.yellow_goal = torch.tensor(
            [self.field_width / 2, 0.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_vss_reward_and_dones(
    ball_pos,
    robot_pos,
    actions,
    reset_buf,
    progress_buf,
    yellow_goal,
    max_episode_length,
    field_lenght,
    goal_width,
    goal_w,
    move_w,
    grad_w,
    energy_w,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    # Negative what we want to reduce, Positive what we want to increase

    zeros = torch.zeros_like(reset_buf)
    ones = torch.ones_like(reset_buf)

    # GOAL
    is_goal = (torch.abs(ball_pos[:, 0]) > (field_lenght / 2)) & (
        torch.abs(ball_pos[:, 1]) < (goal_width / 2)
    )
    is_goal_blue = is_goal & (ball_pos[..., 0] > 0)
    is_goal_yellow = is_goal & (ball_pos[..., 0] < 0)
    goal_rw = torch.where(is_goal_blue, ones, zeros)
    goal_rw = -torch.where(is_goal_yellow, ones, goal_rw)

    # MOVE
    dist_robot_ball = -torch.norm(robot_pos - ball_pos, dim=1)

    # GRAD
    dist_ball_left_goal = torch.norm(ball_pos - (-yellow_goal), dim=1)
    dist_ball_right_goal = -torch.norm(ball_pos - yellow_goal, dim=1)

    # ENERGY
    energy = torch.sum(torch.abs(actions), dim=1)

    reward = (
        goal_rw * goal_w
        + dist_robot_ball * move_w
        + (dist_ball_left_goal + dist_ball_right_goal) * grad_w
        + energy * energy_w
    )

    ones = torch.ones_like(reset_buf)
    reset = torch.zeros_like(reset_buf)

    reset = torch.where(is_goal, ones, reset)
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)

    return reward, reset
