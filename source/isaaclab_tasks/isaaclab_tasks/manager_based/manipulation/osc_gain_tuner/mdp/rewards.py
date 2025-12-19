# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def ee_linear_velocity_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large end-effector linear speeds (encourage smoothness / slower motion).

    Returns L2 norm of the end-effector linear velocity for each environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # asset.data.body_vel_w shape: (num_envs, 3, num_bodies) or (num_envs, num_bodies, 3) depending on API;
    # the common Isaac/IsaacLab layout is (num_envs, 3, num_bodies) -> pick column accordingly.
    # We try the most common layout and fallback if needed.
    try:
        vel = asset.data.body_vel_w[:, :, asset_cfg.body_ids[0]]  # shape (N, 3)
    except Exception:
        # try alternative layout
        vel = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :]
    return torch.norm(vel, dim=1)


def ee_angular_velocity_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large end-effector angular speeds."""
    asset: RigidObject = env.scene[asset_cfg.name]
    try:
        ang_vel = asset.data.body_ang_vel_w[:, :, asset_cfg.body_ids[0]]  # shape (N, 3)
    except Exception:
        ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    return torch.norm(ang_vel, dim=1)


def stiffness_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Small penalty on commanded stiffness magnitude (if gains are exposed via actions).

    This implementation assumes the *last action* contains a stiffness channel (or multiple channels).
    We make a safe attempt to extract stiffness from env.action_manager or env.latest_action data structures.
    If your environment stores the last action differently, adapt this function accordingly.
    """
    # default: zero penalty if extraction fails, but try common locations
    try:
        last_action = env.action_manager.get_last_action()  # hypothetical API
    except Exception:
        # fallback (many IsaacLab envs expose last actions in env._last_actions or env._last_action)
        last_action = getattr(env, "_last_actions", None) or getattr(env, "_last_action", None)

    if last_action is None:
        # nothing to penalize (return zeros)
        return torch.zeros(env.num_envs, device=env.device)

    # locate stiffness channel(s) in the action vector:
    # heuristic: assume last channels correspond to stiffness/damping when stiffness_scale set in action cfg.
    # We attempt to extract the last 1 or 2 channels as stiffness/damping.
    if isinstance(last_action, torch.Tensor):
        act = last_action
    else:
        act = torch.as_tensor(last_action, device=env.device)

    # assume shape (N, A); take last channel(s)
    if act.ndim == 1:
        act = act.unsqueeze(0)
    stiffness_channel = act[:, -2] if act.shape[1] >= 2 else act[:, -1]
    # penalty = L2 of stiffness channel
    return torch.abs(stiffness_channel)  # absolute value is a simple regularizer


def damping_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Small penalty on commanded damping magnitude (if gains are exposed via actions)."""
    try:
        last_action = env.action_manager.get_last_action()
    except Exception:
        last_action = getattr(env, "_last_actions", None) or getattr(env, "_last_action", None)

    if last_action is None:
        return torch.zeros(env.num_envs, device=env.device)

    if isinstance(last_action, torch.Tensor):
        act = last_action
    else:
        act = torch.as_tensor(last_action, device=env.device)
    if act.ndim == 1:
        act = act.unsqueeze(0)
    damping_channel = act[:, -1]  # assume last channel is damping ratio if at least 1 extra channel exists
    return torch.abs(damping_channel)