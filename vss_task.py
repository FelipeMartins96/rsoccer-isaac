from itertools import combinations
import os
from typing import Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from torch import Tensor
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis, get_euler_xyz

BLUE = gymapi.Vec3(0.0, 0.0, 1.0)
YELLOW = gymapi.Vec3(1.0, 1.0, 0.0)

RED = gymapi.Vec3(1.0, 0.0, 0.0)
GREEN = gymapi.Vec3(0.0, 1.0, 0.0)
PURPLE = gymapi.Vec3(1.0, 0.0, 1.0)
CYAN = gymapi.Vec3(0.0, 1.0, 1.0)

TEAM_COLORS = [BLUE, YELLOW]
ID_COLORS = [RED, GREEN, PURPLE, CYAN]


def get_cfg():
    cfg = {
        'rl_device': 'cuda:0',
        'sim_device': 'cuda:0',
        'graphics_device_id': 0,
        'headless': False,
        'virtual_screen_capture': True,
        'force_render': True,
        'physics_engine': 'physx',
    }

    cfg['env'] = {
        'numEnvs': 2048,
    }

    cfg['sim'] = {
        'use_gpu_pipeline': True,
        'up_axis': 'z',
        'dt': 1 / 20,
        'gravity': [0, 0, -9.81],
        'substeps': 3,
    }

    cfg['sim']['physx'] = {
        'use_gpu': True,
        'bounce_threshold_velocity': 0.1,
        'contact_offset': 0.01,
        'max_depenetration_velocity': 10.0,
    }

    return cfg


class VSS3v3SelfPlay(VecTask):
    def __init__(self, has_grad=True, record=False, num_envs=2048):
        self.cfg = get_cfg()
        self.cfg['env']['numEnvs'] = num_envs
        if record:
            self.cfg['virtual_screen_capture'] = True
            self.cfg['force_render'] = False
            self.cfg['headless'] = False
        else:
            self.cfg['virtual_screen_capture'] = False
            self.cfg['force_render'] = False
            self.cfg['headless'] = True

        self.max_episode_length = 400

        self.n_robots_per_team = 3
        self.n_robot_dofs = 2

        # FIXED VALUES
        self.n_teams = 2
        self.n_balls = 1  # does not support more

        self.n_robots = self.n_robots_per_team * self.n_teams

        self.env_total_width = 2
        self.env_total_height = 1.5
        self.robot_max_wheel_rad_s = 42.0
        self.field_width = 1.5
        self.field_height = 1.3
        self.goal_height = 0.4
        self.min_dist = 0.07

        self.w_goal = 1
        self.w_grad = 2 if has_grad else 0

        self.n_agents = self.n_teams
        self.n_team_actions = self.n_robots_per_team * self.n_robot_dofs
        self.cfg['env']['numAgents'] = self.n_agents
        self.cfg['env']['numActions'] = self.n_team_actions * self.n_teams
        self.cfg['env']['numObservations'] = (
            4 + (self.n_robots) * 7 + self.n_team_actions
        )

        self.cfg['virtual_screen_capture'] = record

        super().__init__(
            config=self.cfg,
            rl_device=self.cfg['rl_device'],
            sim_device=self.cfg['sim_device'],
            graphics_device_id=self.cfg['graphics_device_id'],
            headless=self.cfg['headless'],
            virtual_screen_capture=self.cfg['virtual_screen_capture'],
            force_render=self.cfg['force_render'],
        )

        entities_ids = list(range(self.n_robots + self.n_balls))
        self.combinations = torch.tensor(
            list(combinations(entities_ids, 2)), device=self.device
        )

        self._acquire_tensors()
        self._refresh_tensors()
        self.reset_dones()
        self.compute_observations()

        if self.viewer != None:
            cam_pos = gymapi.Vec3(2, 0.0, 5)
            cam_target = gymapi.Vec3(2, 1.0, 0.0)
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
            # Blue Robots
            for j in range(self.n_robots_per_team):
                self._add_robot(_env, i, 0, j)
            # Yellow Robots
            for j in range(self.n_robots_per_team):
                self._add_robot(_env, i, 1, j)

            self._add_field(_env, i)

    def pre_physics_step(self, _actions):
        # reset progress_buf for envs reseted on previous step
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.progress_buf[env_ids] = 0

        self.dof_velocity_buf[:] = _actions.to(self.device)

        act = self.dof_velocity_buf * self.robot_max_wheel_rad_s
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(act))

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_rewards_and_dones()

        # Save observations previously to resets
        self.compute_observations()
        self.extras["terminal_observation"] = self.obs_buf.clone().to(self.rl_device)
        self.extras["terminal_rewards"] = {
            "goal": self.rw_goal.clone().to(self.rl_device),
            "grad": self.rw_grad.clone().to(self.rl_device),
        }
        self.extras["progress_buffer"] = (
            self.progress_buf.clone().to(self.rl_device).float()
        )

        self.reset_dones()
        self.compute_observations()

    def compute_rewards_and_dones(self):
        # goal, grad
        _, p_grad = compute_vss_rewards(
            self.ball_pos,
            self.rew_buf[:, 0],
            self.yellow_goal,
            self.field_width,
            self.goal_height,
        )
        self._refresh_tensors()
        goal, grad = compute_vss_rewards(
            self.ball_pos,
            self.rew_buf[:, 0],
            self.yellow_goal,
            self.field_width,
            self.goal_height,
        )

        goal_rw = self.w_goal * goal
        grad_rw = self.w_grad * (grad - p_grad)

        self.rw_goal += goal_rw
        self.rw_grad += grad_rw

        rw_sum = (goal_rw + grad_rw).view(-1, 1)

        self.rew_buf[:] = torch.cat((rw_sum, -rw_sum), dim=1)

        self.reset_buf = compute_vss_dones(
            ball_pos=self.ball_pos,
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            max_episode_length=self.max_episode_length,
            field_width=self.field_width,
            goal_height=self.goal_height,
        )

    def compute_observations(self):
        self.obs_buf[:, 0, :2] = self.ball_pos
        self.obs_buf[:, 0, 2:4] = self.ball_vel
        self.obs_buf[:, 0, 4 : -self.n_team_actions] = compute_robots_obs(
            self.robots_pos,
            self.robots_vel,
            self.robots_quats,
            self.robots_ang_vel,
            self.n_robots,
            self.num_envs,
        )
        self.obs_buf[:, 0, -self.n_team_actions :] = self.dof_velocity_buf[
            ..., : self.n_team_actions
        ]

        # getting yellow obs by mirroring blue obs
        self.obs_buf[:, 1, :4] = -self.obs_buf[:, 0, :4]
        mirror_obs = (
            self.obs_buf[:, 0, 4 : -self.n_team_actions].view(
                self.num_envs, self.n_robots, -1
            )
            * self.mirror_tensor
        )
        yellow_obs = self.obs_buf[:, 1, 4 : -self.n_team_actions].view(
            self.num_envs, self.n_robots, -1
        )
        yellow_obs[:, : self.n_robots_per_team, :] = mirror_obs[
            :, self.n_robots_per_team :, :
        ]
        yellow_obs[:, self.n_robots_per_team :, :] = mirror_obs[
            :, : self.n_robots_per_team, :
        ]
        self.obs_buf[:, 1, -self.n_team_actions :] = self.dof_velocity_buf[
            ..., -self.n_team_actions :
        ]

    def reset_dones(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            # Reset env state
            self.root_state[env_ids] = self.env_reset_root_state

            # randomize positions
            rand_pos = (
                torch.rand(
                    (len(env_ids), self.n_robots + self.n_balls, 2),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                - 0.5
            ) * self.field_scale

            # Check for colliding robots
            dists = torch.linalg.norm(
                rand_pos[:, self.combinations[:, 0], :]
                - rand_pos[:, self.combinations[:, 1], :],
                dim=2,
            )
            too_close = torch.any(dists < self.min_dist, dim=1)
            close_ids = too_close.nonzero(as_tuple=False).squeeze(-1)
            while len(close_ids):
                rand_pos[close_ids] = (
                    torch.rand(
                        (len(close_ids), self.n_robots + self.n_balls, 2),
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    - 0.5
                ) * self.field_scale
                dists = torch.linalg.norm(
                    rand_pos[:, self.combinations[:, 0], :]
                    - rand_pos[:, self.combinations[:, 1], :],
                    dim=2,
                )
                too_close = torch.any(dists < self.min_dist, dim=1)
                close_ids = too_close.nonzero(as_tuple=False).squeeze(-1)

            self.ball_pos[env_ids] = rand_pos[:, self.ball]
            self.robots_pos[env_ids] = rand_pos[:, self.robots]

            # randomize rotations
            rand_angles = torch_rand_float(
                -np.pi, np.pi, (len(env_ids), self.n_robots), device=self.device
            )
            self.robots_quats[env_ids] = quat_from_angle_axis(rand_angles, self.z_axis)

            # randomize ball velocities
            rand_ball_vel = (
                torch.rand(
                    (len(env_ids), 2),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                - 0.5
            ) * 1
            self.ball_vel[env_ids] = rand_ball_vel

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_state)
            )

            self.rw_goal[env_ids] = 0.0
            self.rw_grad[env_ids] = 0.0
            self.dof_velocity_buf[env_ids] *= 0.0

    def allocate_buffers(self):
        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_agents, self.num_obs),
            device=self.device,
            dtype=torch.float,
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_agents, self.num_states),
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(
            (self.num_envs, self.num_agents),
            device=self.device,
            dtype=torch.float,
        )
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.extras = {}

    def _add_ground(self):
        pp = gymapi.PlaneParams()
        pp.distance = 0.0
        pp.dynamic_friction = 0.4
        pp.normal = gymapi.Vec3(
            0, 0, 1
        )  # defaultgymapi.Vec3(0.000000, 1.000000, 0.000000)
        pp.restitution = 0.0
        pp.segmentation_id = 0
        pp.static_friction = 0.7
        self.gym.add_ground(self.sim, pp)

    def _add_ball(self, env, env_id):
        options = gymapi.AssetOptions()
        options.density = 1130.0  # 0.046 kg
        color = gymapi.Vec3(1.0, 0.4, 0.0)
        radius = 0.02134
        pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, radius))
        asset = self.gym.create_sphere(self.sim, radius, options)
        ball = self.gym.create_actor(
            env=env, asset=asset, pose=pose, group=env_id, filter=0b01, name='ball'
        )
        self.gym.set_rigid_body_color(env, ball, 0, gymapi.MESH_VISUAL, color)

    def _add_robot(self, env, env_id, team, idx):
        options = gymapi.AssetOptions()
        root = os.path.dirname(os.path.abspath(__file__))
        rbt_asset = self.gym.load_asset(
            sim=self.sim,
            rootpath=root,
            filename='assets/vss_robot.urdf',
            options=options,
        )
        body, left_wheel, right_wheel, tag_id, tag_team = 0, 1, 2, 3, 4
        initial_height = 0.024  # _z dimension
        pos_idx = idx + 1 if team else -(idx + 1)
        pose = gymapi.Transform(p=gymapi.Vec3(0.1 * pos_idx, 0.0, initial_height))
        robot = self.gym.create_actor(
            env=env, asset=rbt_asset, pose=pose, group=env_id, filter=0b00, name='robot'
        )
        self.gym.set_rigid_body_color(
            env, robot, tag_id, gymapi.MESH_VISUAL, TEAM_COLORS[team]
        )
        self.gym.set_rigid_body_color(
            env, robot, tag_team, gymapi.MESH_VISUAL, ID_COLORS[idx]
        )
        props = self.gym.get_actor_rigid_shape_properties(env, robot)
        props[body].friction = 0.0
        props[body].filter = 0b0
        props[left_wheel].filter = 0b11
        props[left_wheel].friction = 0.7
        props[right_wheel].filter = 0b11
        props[right_wheel].friction = 0.7
        self.gym.set_actor_rigid_shape_properties(env, robot, props)

        props = self.gym.get_actor_dof_properties(env, robot)
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.01)
        props['armature'].fill(0.0002)
        props['friction'].fill(0.0002)
        props['velocity'].fill(self.robot_max_wheel_rad_s)
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

        self.field_scale = (
            torch.tensor(
                [self.field_width, self.field_height],
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            - self.min_dist
        )

        n_field_actors = 8  # 2 side walls, 4 end walls, 2 goal walls
        self.num_actors = self.n_balls + self.n_robots + n_field_actors
        self.ball = 0
        self.robots = slice(self.n_balls, self.n_balls + self.n_robots)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_state).view(
            self.num_envs, self.num_actors, 13
        )

        self.root_pos = self.root_state[..., 0:2]
        self.robots_pos = self.root_pos[:, self.robots, :]
        self.ball_pos = self.root_pos[:, self.ball, :]

        self.root_quats = self.root_state[..., 3:7]
        self.robots_quats = self.root_quats[:, self.robots, :]

        self.root_vel = self.root_state[..., 7:9]
        self.robots_vel = self.root_vel[:, self.robots, :]
        self.ball_vel = self.root_vel[:, self.ball, :]

        self.root_ang_vel = self.root_state[..., 12]
        self.robots_ang_vel = self.root_ang_vel[:, self.robots]

        self._refresh_tensors()
        self.env_reset_root_state = self.root_state[0].clone()
        self.z_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.yellow_goal = torch.tensor(
            [self.field_width / 2, 0.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.rw_goal = torch.zeros_like(
            self.rew_buf[:, 0], device=self.device, requires_grad=False
        )
        self.rw_grad = torch.zeros_like(
            self.rew_buf[:, 0], device=self.device, requires_grad=False
        )
        self.dof_velocity_buf = torch.zeros(
            (self.num_envs, self.n_robots * 2), device=self.device, requires_grad=False
        )
        self.mirror_tensor = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
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
def compute_vss_rewards(ball_pos, rew_buf, yellow_goal, field_width, goal_height):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    # Negative what we want to reduce, Positive what we want to increase

    zeros = torch.zeros_like(rew_buf)
    ones = torch.ones_like(rew_buf)

    # GOAL (yellow_goal = -1, no_goal = 0, blue_goal = 1)
    is_goal = (torch.abs(ball_pos[:, 0]) > (field_width / 2)) & (
        torch.abs(ball_pos[:, 1]) < (goal_height / 2)
    )
    is_goal_blue = is_goal & (ball_pos[..., 0] > 0)
    is_goal_yellow = is_goal & (ball_pos[..., 0] < 0)
    goal = torch.where(is_goal_blue, ones, zeros)
    goal = torch.where(is_goal_yellow, -ones, goal)

    # GRAD
    dist_ball_left_goal = torch.norm(ball_pos - (-yellow_goal), dim=1)
    dist_ball_right_goal = torch.norm(ball_pos - yellow_goal, dim=1)
    grad = dist_ball_left_goal - dist_ball_right_goal

    # goal, grad
    return goal, grad


@torch.jit.script
def compute_vss_dones(
    ball_pos,
    reset_buf,
    progress_buf,
    max_episode_length,
    field_width,
    goal_height,
):
    # type: (Tensor, Tensor, Tensor, float, float, float) -> Tensor

    # CHECK GOAL
    is_goal = (torch.abs(ball_pos[:, 0]) > (field_width / 2)) & (
        torch.abs(ball_pos[:, 1]) < (goal_height / 2)
    )

    ones = torch.ones_like(reset_buf)
    reset = torch.zeros_like(reset_buf)

    reset = torch.where(is_goal, ones, reset)
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)

    return reset


@torch.jit.script
def compute_robots_obs(
    robots_pos,
    robots_vel,
    robots_quats,
    robots_ang_vel,
    n_robots,
    num_envs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, int, int) -> Tensor
    robots_obs = torch.zeros(
        (num_envs, n_robots, 7), dtype=torch.float, device=robots_pos.device
    )
    robots_obs[..., 0:2] = robots_pos
    robots_obs[..., 2:4] = robots_vel
    _, _, angles = get_euler_xyz(robots_quats.reshape(-1, 4))
    angles = angles.reshape(num_envs, n_robots)
    robots_obs[..., 4] = torch.cos(angles)
    robots_obs[..., 5] = torch.sin(angles)
    robots_obs[..., 6] = robots_ang_vel / 50.0

    return robots_obs.view(num_envs, -1)
