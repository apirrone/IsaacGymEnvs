# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pickle
import time
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import calc_heading_quat_inv


class BdxAMPBase(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]

        self.common_step_counter = 0
        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"][
            "linearVelocityXYRewardScale"
        ]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"][
            "angularVelocityZRewardScale"
        ]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.push_robots = self.cfg["task"]["push_robots"]
        self.push_robots_interval = self.cfg["task"]["push_robots_params"]["interval"]
        self.max_push_force = self.cfg["task"]["push_robots_params"]["max_force"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.debug_save_obs_actions = self.cfg["env"]["debugSaveObsActions"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"][
            "linear_x"
        ]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"][
            "linear_y"
        ]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 55
        # self.cfg["env"]["numObservations"] = 54 # lin vel
        self.cfg["env"]["numActions"] = 15

        # Call super init earlier to initialize sim params
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        # _imu_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

        self.motion_ids = None

        self.command_ranges = {}
        self.command_ranges["bdx_walk_forward"] = {}
        self.command_ranges["bdx_walk_forward"]["x"] = [0.0, 0.02]
        self.command_ranges["bdx_walk_forward"]["y"] = [0.0, 0.0]
        self.command_ranges["bdx_walk_forward"]["yaw"] = [0.0, 0.0]

        self.command_ranges["bdx_walk_backward"] = {}
        self.command_ranges["bdx_walk_backward"]["x"] = [-0.02, 0.0]
        self.command_ranges["bdx_walk_backward"]["y"] = [0.0, 0.0]
        self.command_ranges["bdx_walk_backward"]["yaw"] = [0.0, 0.0]

        self.command_ranges["bdx_walk_left"] = {}
        self.command_ranges["bdx_walk_left"]["x"] = [0.0, 0.0]
        self.command_ranges["bdx_walk_left"]["y"] = [0.0, 0.02]
        self.command_ranges["bdx_walk_left"]["yaw"] = [0.0, 0.0]

        self.command_ranges["bdx_walk_right"] = {}
        self.command_ranges["bdx_walk_right"]["x"] = [0.0, 0.0]
        self.command_ranges["bdx_walk_right"]["y"] = [-0.02, 0.0]
        self.command_ranges["bdx_walk_right"]["yaw"] = [0.0, 0.0]

        self.command_ranges["bdx_turn_left"] = {}
        self.command_ranges["bdx_turn_left"]["x"] = [0.0, 0.0]
        self.command_ranges["bdx_turn_left"]["y"] = [0.0, 0.0]
        self.command_ranges["bdx_turn_left"]["yaw"] = [0.0, 0.05]

        self.command_ranges["bdx_turn_right"] = {}
        self.command_ranges["bdx_turn_right"]["x"] = [0.0, 0.0]
        self.command_ranges["bdx_turn_right"]["y"] = [0.0, 0.0]
        self.command_ranges["bdx_turn_right"]["yaw"] = [-0.05, 0.0]

        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.default_dof_vel = torch.zeros_like(
            self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False
        )
        # self.imu_indices = torch.tensor(
        #     [3, 4, 5], device=self.device, requires_grad=False
        # )
        # self.imu_tensor = gymtorch.wrap_tensor(_imu_tensor)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(
            self.base_init_state, device=self.device, requires_grad=False
        )
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._build_pd_action_offset_scale()

        if self.viewer != None:
            self._init_camera()

        # _imu_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # # self.imu_indices = torch.tensor(
        # #     [3, 4, 5], device=self.device, requires_grad=False
        # # )
        # self.imu_indices = torch.tensor(
        #     [0, 1, 2], device=self.device, requires_grad=False
        # )  # TODO what indices ?
        # self.imu_tensor = gymtorch.wrap_tensor(_imu_tensor)
        #
        if self.debug_save_obs_actions:
            self.saved_obs = []
            self.saved_actions = []

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        asset_file = "urdf/bdx/urdf/bdx.urdf"
        # asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 1
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        bdx_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(bdx_asset)

        self.num_bodies = self.gym.get_asset_rigid_body_count(bdx_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(bdx_asset)
        self.dof_names = self.gym.get_asset_dof_names(bdx_asset)

        # WARNING TODO
        # This is the joints order when loading using IsaacGymEnvs
        # ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
        # This is the "standard" order (from mujoco)
        # ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle', 'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna']

        feet_names = ["right_foot", "left_foot"]
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )

        knee_names = ["right_knee", "left_knee"]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )

        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(bdx_asset)

        # # add imu sensor
        # body_idx = self.gym.find_asset_rigid_body_index(bdx_asset, "body_module")
        # sensor_pose = gymapi.Transform()  # gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        # self.gym.create_asset_force_sensor(bdx_asset, body_idx, sensor_pose)

        self.dof_limits_lower = []
        self.dof_limits_upper = []
        # for i in range(self.num_dof):
        #     dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
        #     dof_props["stiffness"][i] = 0
        #     dof_props["damping"][i] = 0
        #     if dof_props["lower"][i] > dof_props["upper"][i]:
        #         self.dof_limits_lower.append(dof_props["upper"][i])
        #         self.dof_limits_upper.append(dof_props["lower"][i])
        #     else:
        #         self.dof_limits_lower.append(dof_props["lower"][i])
        #         self.dof_limits_upper.append(dof_props["upper"][i])

        # Previously
        for i in range(self.num_dof):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = self.cfg["env"]["control"][
                "stiffness"
            ]  # self.Kp
            dof_props["damping"][i] = self.cfg["env"]["control"]["damping"]  # self.Kd
            if dof_props["lower"][i] > dof_props["upper"][i]:
                self.dof_limits_lower.append(dof_props["upper"][i])
                self.dof_limits_upper.append(dof_props["lower"][i])
            else:
                self.dof_limits_lower.append(dof_props["lower"][i])
                self.dof_limits_upper.append(dof_props["upper"][i])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.bdx_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            bdx_handle = self.gym.create_actor(
                env_ptr, bdx_asset, start_pose, "bdx", i, 1, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, bdx_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, bdx_handle)
            self.envs.append(env_ptr)
            self.bdx_handles.append(bdx_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.bdx_handles[0], feet_names[i]
            )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.bdx_handles[0], knee_names[i]
            )

        head_name = "head_assembly"
        self.head_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.bdx_handles[0], head_name
        )

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.bdx_handles[0], "base"
        )

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        # self.actions = torch.zeros(
        #     self.num_envs,
        #     self.num_actions,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=False,
        # )
        if self.debug_save_obs_actions:
            self.saved_actions.append((self.actions[0].cpu().numpy(), time.time()))
            pickle.dump(self.saved_actions, open("saved_actions.pkl", "wb"))

        if self._pd_control:
            # target = self._action_to_pd_targets(self.actions) + self.default_dof_pos
            # self.torques = (
            #     self.Kp * (pd_tar - self.dof_pos)
            #     - self.Kd * self.dof_vel  # * self.dof_vel_scale
            # )
            # self.torques = torch.clip(
            #     self.torques, -0.6, 0.6
            # )  # TODO find more restrictive limits based on the walk generator
            # self.gym.set_dof_actuation_force_tensor(
            #     self.sim, gymtorch.unwrap_tensor(self.torques)
            # )

            # pd_target = self._action_to_pd_targets(self.actions) + self.default_dof_pos
            # pd_target = self.default_dof_pos
            # pd_target = (
            #     self.default_dof_pos
            #     + np.sin(5 * self.common_step_counter * self.dt) * 0.5
            # )
            # pd_target = self.actions
            target = self.default_dof_pos + self.actions
            target_tensor = gymtorch.unwrap_tensor(target)
            self.gym.set_dof_position_target_tensor(self.sim, target_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1
        self.common_step_counter += 1

        # self.gym.refresh_force_sensor_tensor(self.sim)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        if self.push_robots:
            if self.common_step_counter % self.push_robots_interval == 0:
                self._push_robots()

        self.compute_observations()
        self.compute_reward(self.actions)
        self.compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

    def compute_reward(self, actions):
        # print(self.motion_ids)

        self.rew_buf[:] = compute_bdx_reward(
            self.root_states,
            self.commands,
            self.rew_scales,
            self.lin_vel_scale,
            self.ang_vel_scale,
        )

    def compute_reset(self):
        self.reset_buf, self._terminate_buf = compute_humanoid_reset(
            self.progress_buf,
            self._terminate_buf,
            self.root_states,
            self.contact_forces,
            self.knee_indices,
            self.base_index,
            self.head_index,
            self.max_episode_length,
            self._termination_height,
            self._enable_early_termination,
        )

    def compute_observations(self, env_ids=None):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # self.gym.refresh_dof_force_tensor(self.sim)
        # imu_state = torch.index_select(self.imu_tensor, 1, self.imu_indices)
        # print(imu_state[0])

        # TODO: Replace default_dof_pos with _pd_action_offset
        if env_ids is None:
            self.obs_buf[:] = compute_bdx_observations(  # tensors
                self.root_states,
                self.dof_pos,
                self.default_dof_pos,
                self.dof_vel,
                self.gravity_vec,
                self.actions,
                self.commands,
                # scales
                self.lin_vel_scale,
                self.ang_vel_scale,
                self.dof_pos_scale,
                self.dof_vel_scale,
            )

        else:
            self.obs_buf[env_ids] = compute_bdx_observations(  # tensors
                self.root_states[env_ids],
                self.dof_pos[env_ids],
                self.default_dof_pos[env_ids],
                self.dof_vel[env_ids],
                self.gravity_vec[env_ids],
                self.actions[env_ids],
                self.commands[env_ids],
                # scales
                self.lin_vel_scale,
                self.ang_vel_scale,
                self.dof_pos_scale,
                self.dof_vel_scale,
            )
        if self.debug_save_obs_actions:
            self.saved_obs.append(self.obs_buf[0].cpu().numpy())
            pickle.dump(self.saved_obs, open("saved_obs.pkl", "wb"))

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)

        # generate relevant commands depending on the choosen motion file
        for i in range(len(env_ids)):
            env_id = env_ids[i]
            motion_file = self._motion_lib._motion_files[self.motion_ids[i]]
            motion_name = motion_file.split("/")[-1].split(".")[0]
            command_range = self.command_ranges[motion_name]

            self.commands_x[env_id] = torch_rand_float(
                command_range["x"][0],
                command_range["x"][1],
                (len([env_id]), 1),
                device=self.device,
            ).squeeze()
            self.commands_y[env_id] = torch_rand_float(
                command_range["y"][0],
                command_range["y"][1],
                (len([env_id]), 1),
                device=self.device,
            ).squeeze()
            self.commands_yaw[env_id] = torch_rand_float(
                command_range["yaw"][0],
                command_range["yaw"][1],
                (len([env_id]), 1),
                device=self.device,
            ).squeeze()

        self.compute_observations(env_ids)

    def _reset_actors(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if self.debug_save_obs_actions:
            self.saved_obs = []
            self.saved_actions = []

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = self.default_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._terminate_buf[env_ids] = 0

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2]
        )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

    def render(self, mode="rgb_array"):
        if self.viewer and self.camera_follow:
            self._update_camera()

        return super().render(mode=mode)

    def _build_pd_action_offset_scale(self):
        num_joints = 15

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for dof_offset in range(num_joints):
            curr_low = lim_low[dof_offset]
            curr_high = lim_high[dof_offset]
            curr_mid = 0.5 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = 0.7 * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[dof_offset] = curr_low
            lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        print("Push !")
        max_vel = self.max_push_force
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor

    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_bdx_reward(
    # tensors
    root_states,
    commands,
    # Dict
    rew_scales,
    # Other
    lin_vel_scale,
    ang_vel_scale,
):
    # type: (Tensor, Tensor, Dict[str, float], float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    # base_lin_vel = root_states[:, 7:10] * lin_vel_scale
    # base_ang_vel = root_states[:, 10:13] * ang_vel_scale
    # velocity tracking reward
    lin_vel_error = torch.sum(
        torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1
    )

    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * rew_scales["ang_vel_z"]
    total_reward = rew_lin_vel_xy + rew_ang_vel_z
    total_reward = torch.clip(total_reward, 0.0, None)

    return total_reward


@torch.jit.script
def compute_humanoid_reset(
    # tensors
    reset_buf,
    progress_buf,
    root_states,
    contact_forces,
    knee_indices,
    # other
    base_index,
    head_index,
    max_episode_length,
    termination_height,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int, int, int, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    if enable_early_termination:
        # terminated = terminated | (torch.norm(contact_forces[:, base_index, :], dim=1) > 1.)
        # terminated = terminated | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)

        # head contact
        # terminated = terminated | (torch.norm(contact_forces[:, head_index, :], dim=1) > 1.0)
        body_height = root_states[:, 2]
        terminated = terminated | (body_height < termination_height)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )
    return reset, terminated


@torch.jit.script
def compute_bdx_observations(
    root_states,
    dof_pos,
    default_dof_pos,
    dof_vel,
    gravity_vec,
    actions,
    commands,
    lin_vel_scale,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    heading_rot = calc_heading_quat_inv(base_quat)
    local_base_quat = quat_mul(heading_rot, base_quat)

    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    # projected_gravity = quat_rotate(base_quat, gravity_vec)
    # base_lin_vel = root_states[:, 7:10] * lin_vel_scale
    # base_ang_vel = root_states[:, 10:13] * ang_vel_scale
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    dof_vel = dof_vel * dof_vel_scale
    obs = torch.cat(
        (
            local_base_quat,
            # base_lin_vel,
            base_ang_vel,
            dof_pos_scaled,
            dof_vel,
            actions,
            commands,
        ),
        dim=-1,
    )

    return obs
