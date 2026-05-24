# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import os
import sys
from typing import List, Tuple, Union

sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import joblib
import re
import pandas as pd
from pathlib import Path

from scipy.spatial.transform import Rotation as sRot
import mujoco
import random
random.seed(0)
from src.utils.torch_utils import to_torch
from easydict import EasyDict
import scipy.ndimage as ndimage

# ── FK tracking: bodies tracked by the reward and policy observation ──────────
# Order is fixed — index 0 = pelvis (gets separate weight), indices 1-7 = limbs
TRACKED_BODY_NAMES = [
    'pelvis',                           # [0] pelvis — separate weight
    'femur_r', 'femur_l',               # [1,2] thighs
    'tibia_r', 'tibia_l',               # [3,4] tibias
    'calcn_l',                          # [5]   left calcaneus (heel)
    'toes_l',                           # [6]   left toes (intact foot)
]
N_TRACKED_BODIES = len(TRACKED_BODY_NAMES)  # 7

torch.set_num_threads(1)

class ProstWalkCore:

    def __init__(self, config, joint_names=None, mj_model=None):
        self.config = config
        self.dtype = np.float32
        self.joint_names = joint_names
        self._mj_model = mj_model

        self.load_data(config.motion_file)
        self._curr_motion_ids = np.arange(self._num_unique_motions)
        self._sampling_batch_prob = (
            np.ones(self._num_unique_motions) / self._num_unique_motions
        )
        self._velocity_groups = self._bunch_by_velocity()

    def load_data(self, filepath: str) -> None:
        """
        Loads motion data from a given pickle file or a directory of OpenSim .mot files.
        """
        if os.path.isdir(filepath):
            self.motion_data = self._load_opensim_dir(filepath)
        else:
            self.motion_data = joblib.load(filepath)
        self._num_unique_motions = len(self.motion_data.keys())
        self._curr_motion_ids = np.array(list(range(self._num_unique_motions)))

    def _load_opensim_dir(self, directory: str) -> dict:
        """Loads all .mot files in a directory and returns a dictionary of motion data."""
        cache_file = os.path.join(directory, "processed_motions.joblib")
        if os.path.exists(cache_file):
            print(f"Loading cached OpenSim data from {cache_file}")
            return joblib.load(cache_file)
            
        motion_data = {}
        mot_files = list(Path(directory).glob("*.mot"))
        print(f"Parsing {len(mot_files)} .mot files...")
        for mot_file in tqdm(mot_files):
            name = mot_file.stem
            data = self._parse_mot(str(mot_file), self.joint_names)
            motion_data[name] = data
        
        # Cache for next time
        print(f"Caching processed data to {cache_file}")
        joblib.dump(motion_data, cache_file)
        return motion_data

    def _parse_mot(self, filepath: str, mu_joint_names: List[str] = None) -> dict:
        """Parses an OpenSim .mot file into a dictionary compatible with ProstWalkCore."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_end = 0
        in_degrees = True
        for i, line in enumerate(lines):
            if 'inDegrees' in line:
                in_degrees = 'yes' in line.lower()
            if 'endheader' in line:
                header_end = i + 1
                break
        
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_end)
        
        # Minimally map Pelvis and Time
        fps = 1.0 / (df['time'].iloc[1] - df['time'].iloc[0]) if len(df) > 1 else 30
        
        # Detect units
        unit_scale = np.pi / 180.0 if in_degrees else 1.0
        
        # ============================================================
        # Frame rotation: OpenSim Y-up -> MuJoCo Z-up
        # DISABLED: The XML model uses a body-level quat on the pelvis
        # to handle Y-up -> Z-up visually. Joint qpos values remain in
        # the OpenSim Y-up local frame, so no rotation is needed here.
        # ============================================================
        # R_y_neg = sRot.from_euler('y', -np.pi/2)
        # R_x_pos = sRot.from_euler('x', np.pi/2)
        # R_frame = R_x_pos
        # R_frame_matrix = R_frame.as_matrix()  # 3x3
        
        # --- Pelvis Translation (raw OpenSim Y-up, matches model local frame) ---
        pelvis_trans_raw = df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].values
        pelvis_trans = pelvis_trans_raw  # No rotation needed
        # pelvis_trans = (R_frame_matrix @ pelvis_trans_raw.T).T  # (N, 3) — old rotated version
        
        # --- Pelvis Orientation (raw OpenSim Euler angles) ---
        pelvis_euler = df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].values * unit_scale
        # Old rotated version:
        # pelvis_rot_opensim = sRot.from_euler('xyz', pelvis_euler)
        # pelvis_rot_mujoco = R_frame * pelvis_rot_opensim
        # pelvis_quat = pelvis_rot_mujoco.as_quat()  # [x, y, z, w]
        
        if mu_joint_names is not None:
             # Identify actual hinges/slides vs the root joints
             # For OSL_A, root joints are pelvis_tx/ty/tz and pelvis_tilt/list/rotation
             root_names = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
             real_joint_names = [n for n in mu_joint_names if n not in root_names and 'root' not in n.lower()]
             joint_angles = np.zeros((len(df), len(real_joint_names)), dtype=np.float32)
             
             # Name mapping for OSL joints (Expert .mot vs MuJoCo model)
             name_map = {
                 'osl_ankle_angle_r': 'ankle_angle_r',
                 'osl_knee_angle_r': 'knee_angle_r'
             }
             
             for i, mu_name in enumerate(real_joint_names):
                  # Check for direct match or mapped match
                  col_name = name_map.get(mu_name, mu_name)
                  if col_name in df.columns:
                       val = df[col_name].values * unit_scale
                       # Flip sign for knee joints if needed
                       if col_name in ['knee_angle_r', 'knee_angle_l', 'osl_knee_angle_r']:
                            val = -val
                       joint_angles[:, i] = val
                  else:
                       # Joint not in .mot file (e.g. coupled knee translations, muscle wrapping points).
                       # Use the 'stand' keyframe default value if available, otherwise use joint range midpoint.
                       jnt_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, mu_name)
                       if jnt_id >= 0:
                           qpos_adr = self._mj_model.jnt_qposadr[jnt_id]
                           # Try stand keyframe first
                           stand_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
                           if stand_id >= 0:
                               default_val = self._mj_model.key_qpos[stand_id, qpos_adr]
                           elif self._mj_model.jnt_limited[jnt_id]:
                               # Use midpoint of joint range
                               default_val = 0.5 * (self._mj_model.jnt_range[jnt_id, 0] + self._mj_model.jnt_range[jnt_id, 1])
                           else:
                               default_val = 0.0
                           joint_angles[:, i] = default_val
        else:
            # Fallback to old behavior: extract everything except pelvis
            exclude = ['time', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
            joint_cols = [c for c in df.columns if c not in exclude]
            joint_angles = df[joint_cols].values * unit_scale
        
        # Combine into qpos-like structure for ProstWalkCore
        # Detect if we are using a freejoint (7D) or hinge-based root (6D)
        # myoLeg26_OSL_A uses hinge-based root (pelvis_tx/ty/tz, pelvis_tilt/list/rotation)
        is_hinge_root = 'pelvis_tilt' in mu_joint_names if mu_joint_names else False
        
        if is_hinge_root:
            # Hinge-based root: [tx, ty, tz, tilt, list, rot, joints...]
            # Use raw OpenSim Euler angles directly (no rotation needed)
            qpos = np.concatenate([pelvis_trans, pelvis_euler, joint_angles], axis=1)
        else:
            # Freejoint-based root: [tx, ty, tz, qw, qx, qy, qz, joints...]
            # MuJoCo uses [w, x, y, z]. Need quaternion conversion.
            pelvis_rot_opensim = sRot.from_euler('xyz', pelvis_euler)
            pelvis_quat = pelvis_rot_opensim.as_quat()  # [x, y, z, w]
            pelvis_quat_mj = np.roll(pelvis_quat, 1, axis=1) # [x, y, z, w] -> [w, x, y, z]
            qpos = np.concatenate([pelvis_trans, pelvis_quat_mj, joint_angles], axis=1)

        
        # Compute qvel (from raw OpenSim-frame translation and orientation)
        dt = 1.0 / fps
        # 1. Pelvis linear velocity (OpenSim Y-up local frame)
        lin_vel = np.diff(pelvis_trans, axis=0) / dt
        
        # Extract nominal speed from filename to add to forward velocity
        # Pattern: _0p6_ -> 0.6
        match = re.search(r'_(\d+p\d+)', os.path.basename(filepath))
        if match:
            motion_speed = float(match.group(1).replace('p', '.'))
            lin_vel[:, 0] += motion_speed
        
        # 2. Pelvis angular velocity (OpenSim local frame)
        if is_hinge_root:
            # For hinges, angular velocity is just the diff of Euler angles
            ang_vel = np.diff(pelvis_euler, axis=0) / dt
            # Match length
            ang_vel = np.concatenate([ang_vel, ang_vel[-1:]], axis=0)
        else:
            # For freejoint, convert quaternion diff to angular velocity
            r = sRot.from_quat(pelvis_quat)
            r1 = r[:-1]
            r2 = r[1:]
            # Local angular velocity: r1.inv() * r2
            rel_rot = r1.inv() * r2
            ang_vel_local = rel_rot.as_rotvec() / dt
            # Convert to world frame: omega_world = r1.apply(omega_local)
            ang_vel = r1.apply(ang_vel_local)
            # Match length
            ang_vel = np.concatenate([ang_vel, ang_vel[-1:]], axis=0)
        
        # 3. Joint velocities
        joint_vel = np.diff(joint_angles, axis=0) / dt
        joint_vel = np.concatenate([joint_vel, joint_vel[-1:]], axis=0)
        
        # Match lengths if needed (before concatenation)
        if len(lin_vel) < len(qpos):
             lin_vel = np.concatenate([lin_vel, lin_vel[-1:]], axis=0)
        
        # Combine into qvel-like structure (lin(3), ang(3), joints(N))
        qvel = np.concatenate([lin_vel, ang_vel, joint_vel], axis=1)

        # ── FK body positions and velocities ─────────────────────────────────
        # Requires self._mj_model to be set (passed in __init__ as mj_model=).
        body_xpos, body_vel = self._compute_body_fk(qpos, dt)

        # Store treadmill speed so get_reference_state can accumulate the
        # forward displacement into body_xpos_ref at query time.
        motion_speed = 0.0
        match_speed = re.search(r'_(\d+p\d+)', os.path.basename(filepath))
        if match_speed:
            motion_speed = float(match_speed.group(1).replace('p', '.'))

        return {
            'qpos':        qpos.astype(np.float32),
            'qvel':        qvel.astype(np.float32),
            'fps':         fps,
            'pose_aa':     np.zeros((qpos.shape[0], 24, 3), dtype=np.float32),
            'trans_orig':  pelvis_trans_raw.astype(np.float32),
            'body_xpos':   body_xpos,     # (T, 8, 3)  FK world positions (lab frame, no treadmill offset)
            'body_vel':    body_vel,      # (T, 8, 3)  FK world velocities
            'motion_speed': motion_speed, # m/s — treadmill speed to accumulate in body_xpos_ref at query time
        }

    def _compute_body_fk(self, qpos: np.ndarray, dt: float):
        """
        Run MuJoCo forward kinematics for TRACKED_BODY_NAMES on the given
        qpos sequence.  Returns body_xpos (T, 8, 3) and body_vel (T, 8, 3).

        Velocities are computed via Gaussian-smoothed finite differences,
        matching the approach in Kinesis's convert_opensim.py.
        """
        if self._mj_model is None:
            # Return zeros if no MuJoCo model is available (rare fallback)
            T = qpos.shape[0]
            return (np.zeros((T, N_TRACKED_BODIES, 3), dtype=np.float32),
                    np.zeros((T, N_TRACKED_BODIES, 3), dtype=np.float32))

        # Resolve body IDs once
        body_ids = []
        for name in TRACKED_BODY_NAMES:
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found in MuJoCo model")
            body_ids.append(bid)

        T = qpos.shape[0]
        mj_data = mujoco.MjData(self._mj_model)
        xpos_all = np.zeros((T, N_TRACKED_BODIES, 3), dtype=np.float64)

        for t in range(T):
            mj_data.qpos[:len(qpos[t])] = qpos[t]
            mujoco.mj_kinematics(self._mj_model, mj_data)
            for k, bid in enumerate(body_ids):
                xpos_all[t, k] = mj_data.xpos[bid]

        # Finite-difference velocities (Gaussian smoothed, sigma=2 frames)
        vel_all = np.zeros_like(xpos_all)
        vel_all[1:] = (xpos_all[1:] - xpos_all[:-1]) / dt
        vel_all[0] = vel_all[1]
        vel_all = ndimage.gaussian_filter1d(vel_all, sigma=2, axis=0, mode='nearest')

        return xpos_all.astype(np.float32), vel_all.astype(np.float32)


    def _bunch_by_velocity(self) -> dict:
        """Bunches motion IDs by velocity extracted from keys (filenames)."""
        groups = {}
        for idx, key in enumerate(self.motion_data.keys()):
            # Search for pattern like _0p6 or _1p2
            match = re.search(r'_(\d+p\d+)', key)
            if match:
                vel_str = match.group(1).replace('p', '.')
                vel = float(vel_str)
            else:
                vel = 0.0 # Default/Unknown
            
            if vel not in groups:
                groups[vel] = []
            groups[vel].append(idx)
        return groups

    @property
    def available_speeds(self):
        return sorted(list(self._velocity_groups.keys()))

    def sample_motions_by_velocity(self, velocity: float, n: int = 1) -> np.ndarray:
        """Samples motions specifically from a velocity group."""
        if velocity not in self._velocity_groups:
            print(f"Warning: Velocity {velocity} not found. Sampling from all.")
            return self.sample_motions(n)
        
        group_indices = self._velocity_groups[velocity]
        return np.random.choice(group_indices, size=n, replace=True)

    def load_motions(
            self,
            m_cfg: dict,
            num_motions: int = None,
            shape_params: List[np.ndarray] = None,
            random_sample: bool = True,
            start_idx: int = 0,
            silent: bool = False,
            specific_idxes: np.ndarray = None,
    ):
        
        motions = []
        motion_lengths = []
        motion_fps_acc = []
        motion_dt = []
        motion_num_frames = []
        motion_bodies = []
        motion_aa = []

        self.num_joints = 24

        if num_motions is not None:
             num_motion_to_load = num_motions
        elif shape_params is not None:
             num_motion_to_load = len(shape_params)
        else:
             num_motion_to_load = self._num_unique_motions
        if specific_idxes is not None:
            if len(specific_idxes) < num_motion_to_load:
                num_motion_to_load = len(specific_idxes)
            if random_sample:
                sample_idxes = np.random.choice(
                    specific_idxes,
                    size=num_motion_to_load,
                    replace=False,
                )
            else:
                sample_idxes = specific_idxes
        else:
            if random_sample:
                sample_idxes = np.random.choice(
                    np.arange(self._num_unique_motions),
                    size=num_motion_to_load,
                    replace=False,
                )
            else:
                sample_idxes = np.remainder(
                    np.arange(start_idx, start_idx + num_motion_to_load),
                    self._num_unique_motions,
                )

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = [list(self.motion_data.keys())[i] for i in sample_idxes]
        
        self._sampling_batch_prob = np.ones(len(self._curr_motion_ids)) / len(
            self._curr_motion_ids
        )
        
        motion_data_list = [self.motion_data[self.curr_motion_keys[i]] for i in range(num_motion_to_load)]

        if sys.platform == "darwin":
            num_jobs = 1
        else:
            mp.set_sharing_strategy("file_descriptor")

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(min(mp.cpu_count(), 64), num_motion_to_load)

        if len(motion_data_list) <= 32 or not self.config.multi_thread or num_jobs <= 8:
            num_jobs = 1

        res_acc = {}

        chunk = np.ceil(len(motion_data_list) / num_jobs).astype(int)
        ids = np.arange(len(motion_data_list))

        jobs = [
            (
                ids[i: i + chunk],
                motion_data_list[i: i + chunk],
                self.config,
            )
            for i in range(0, len(motion_data_list), chunk)
        ]
        for i in range(1, len(jobs)):
            worker_args = (*jobs[i], queue, i)
            worker = mp.Process(target=self.load_motions_worker, args=worker_args)
            worker.start()
        res_acc.update(self.load_motions_worker(*jobs[0], None, 0))
        pbar = tqdm(range(len(jobs) - 1))
        for i in pbar:
            res = queue.get()
            res_acc.update(res)
        pbar = tqdm(range(len(res_acc)))

        for f in pbar:
            curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            motion_aa.append(curr_motion.pose_aa)
            motion_fps_acc.append(motion_fps)
            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            motion_lengths.append(curr_len)

            del curr_motion

        self._motion_lengths = np.array(motion_lengths).astype(self.dtype)
        self._motion_fps = np.array(motion_fps_acc).astype(self.dtype)
        self._motion_aa = np.concatenate(motion_aa, axis=0).astype(self.dtype)
        self._motion_dt = np.array(motion_dt).astype(self.dtype)
        self._motion_num_frames = np.array(motion_num_frames)
        self._num_motions = len(motions)

        self.gts = np.concatenate(
            [m.global_translation for m in motions], axis=0
        ).astype(self.dtype)
        self.grs = np.concatenate(
            [m.global_rotation for m in motions], axis=0
        ).astype(self.dtype)
        self.lrs = np.concatenate(
            [m.local_rotation for m in motions], axis=0
        ).astype(self.dtype)
        self.grvs = np.concatenate(
            [m.global_root_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gravs = np.concatenate(
            [m.global_root_angular_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gavs = np.concatenate(
            [m.global_angular_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gvs = np.concatenate([m.global_velocity for m in motions], axis=0).astype(
            self.dtype
        )
        self.dvs = np.concatenate([m.dof_vels for m in motions], axis=0).astype(
            self.dtype
        )
        self.dof_pos = np.concatenate([m.dof_pos for m in motions], axis=0).astype(
            self.dtype
        )
        self.qpos = np.concatenate([m.qpos for m in motions], axis=0).astype(self.dtype)
        self.qvel = np.concatenate([m.qvel for m in motions], axis=0).astype(self.dtype)

        lengths = self._motion_num_frames
        lengths_shifted = np.roll(lengths, 1, axis=0)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = np.arange(len(motions))
        self.num_bodies = self.num_joints

        num_motions = self._num_motions
        total_len = sum(self._motion_lengths)
        print(
            f"###### Sampling {num_motions:d} motions:",
            sample_idxes[:5],
            self.curr_motion_keys[:5],
            f"total length of {total_len:.3f}s and {self.gts.shape[0]} frames.",
        )

        return motions

    def load_motions_worker(
            self,
            ids: np.ndarray,
            motion_data_list: List[dict],
            config: dict,
            queue: Union[mp.Queue, None],
            pid: int,
    ):
        np.random.seed(np.random.randint(5000) * pid)
        res = {}
        for f in range(len(motion_data_list)):
            curr_id = ids[f]
            motion_data = motion_data_list[f]
            
            # Handle OpenSim data that already has qpos/qvel
            if 'qpos' in motion_data and 'qvel' in motion_data:
                fk_motion = EasyDict({
                    'qpos': motion_data['qpos'],
                    'qvel': motion_data['qvel'],
                    'fps': motion_data.get('fps', 30),
                    'global_translation': motion_data.get('global_translation', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                    'global_rotation': motion_data.get('global_rotation', np.zeros((motion_data['qpos'].shape[0], 24, 4))),
                    'local_rotation': motion_data.get('local_rotation', np.zeros((motion_data['qpos'].shape[0], 24, 4))),
                    'global_root_velocity': motion_data['qvel'][:, :3],
                    'global_root_angular_velocity': motion_data['qvel'][:, 3:6],
                    'global_angular_velocity': motion_data.get('global_angular_velocity', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                    'global_velocity': motion_data.get('global_velocity', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                    'dof_pos': motion_data['qpos'][:, 7:],
                    'dof_vels': motion_data['qvel'][:, 6:],
                    'pose_aa': motion_data.get('pose_aa', np.zeros((motion_data['qpos'].shape[0], 24, 3))),
                })
                # Minimally fill global_translation with root pos if missing
                if np.all(fk_motion.global_translation == 0):
                    fk_motion.global_translation[:, 0, :] = fk_motion.qpos[:, :3]
                
                res[curr_id] = fk_motion
                continue

        if queue is not None:
            queue.put(res)
        else:
            return res
            
    def get_motion_state_intervaled(
            self,
            motion_ids,
            motion_times,
            offset=None
    ):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        frame_idx = ((1.0 - blend) * frame_idx0 + blend * frame_idx1).astype(int)
        fl = frame_idx + self.length_starts[motion_ids]

        dof_pos = self.dof_pos[fl]
        body_vel = self.gvs[fl]
        body_ang_vel = self.gavs[fl]
        xpos = self.gts[fl, :]
        xquat = self.grs[fl]
        dof_vel = self.dvs[fl]
        qpos = self.qpos[fl]
        qvel = self.qvel[fl]

        if offset is not None:
            xpos = xpos + offset
            qpos = qpos.copy()
            qpos[..., :3] = qpos[..., :3] + offset

        return EasyDict(
            {
                "root_pos": xpos[..., 0, :].copy(),
                "root_rot": xquat[..., 0, :].copy(),
                "dof_pos": dof_pos.copy(),
                "root_vel": body_vel[..., 0, :].copy(),
                "root_ang_vel": body_ang_vel[..., 0, :].copy(),
                "dof_vel": dof_vel.reshape(dof_vel.shape[0], -1),
                "motion_aa": self._motion_aa[fl],
                "xpos": xpos,
                "xquat": xquat,
                "body_vel": body_vel,
                "body_ang_vel": body_ang_vel,
                # "motion_bodies": self._motion_bodies[motion_ids],
                "qpos": qpos,
                "qvel": qvel,
            }
        )

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def num_all_motions(self) -> int:
        """
        Returns the total number of motions in the dataset.

        Args:
            None

        Returns:
            The total number of motions in the dataset.
        """
        return self._num_unique_motions
    
        
    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.copy()
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0
        frame_idx0 = phase * (num_frames - 1)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)

        blend = np.clip(
            (time - frame_idx0 * dt) / dt, 0.0, 1.0
        )  # clip blend to be within 0 and 1
        return frame_idx0, frame_idx1, blend
    
    def sample_motions(self, n=1):
        # breakpoint()
        motion_ids = np.random.choice(
            np.arange(len(self._curr_motion_ids)),
            size=n,
            p=self._sampling_batch_prob,
            replace=True,
        )
        return motion_ids

    def get_reference_state(
        self,
        motion_key: str,
        t_start: float,
        elapsed: float,
        joint_qpos_idx: list,
        joint_qvel_idx: list,
        subject_settings: dict = None,
    ) -> dict:
        """
        Returns interpolated (q_hat, qdot_hat) for tracked joints at time
        t_start + elapsed within the named motion.

        Reads full qpos/qvel from self.motion_data[motion_key], which is
        already in memory from processed_motions.joblib — no file I/O.

        The reference is CLAMPED at the last frame (no wrap-around).

        The same subject-specific corrections applied during RSI generation
        are re-applied here so that q_hat is consistent with how the agent
        was initialised:
            - pelvis_ty *= height_scale
            - pelvis_list += list_offset_rad
            - osl_ankle_angle_r += ankle_offset_rad
            - pelvis_ty velocity *= height_scale

        Args:
            motion_key:       Key in self.motion_data (e.g. 'tf01_0p6_01_rotated_ik')
            t_start:          Start time in seconds (= frame_idx / fps at reset)
            elapsed:          Seconds elapsed in the current episode
            joint_qpos_idx:   MuJoCo qpos indices for tracked joints (no pelvis)
            joint_qvel_idx:   MuJoCo qvel indices for tracked joints (no pelvis lin vel)
            subject_settings: Dict produced by MyoLegsGAIL._build_subject_settings().
                              Keys: height_scale, list_offset_rad, ankle_offset_rad,
                                    ty_qpos_adr, list_qpos_adr, ankle_r_qpos_adr,
                                    ty_dof_adr

        Returns:
            dict with:
                'q_hat'    — np.ndarray, shape (len(joint_qpos_idx),)
                'qdot_hat' — np.ndarray, shape (len(joint_qvel_idx),)
        """
        if motion_key not in self.motion_data:
            raise KeyError(
                f"motion_key '{motion_key}' not found in motion_data. "
                f"Available keys: {list(self.motion_data.keys())[:5]}..."
            )

        data   = self.motion_data[motion_key]
        qpos_m = data['qpos']          # (T, nq)  full trajectory
        qvel_m = data['qvel']          # (T, nv)
        fps    = float(data['fps'])
        T      = qpos_m.shape[0]

        # Time query — clamp at last frame
        t_query = t_start + elapsed
        frame_f = min(t_query * fps, float(T - 1))
        f0      = int(frame_f)
        f1      = min(f0 + 1, T - 1)
        alpha   = frame_f - f0          # linear blend weight [0, 1)

        # Interpolate full qpos/qvel
        qpos_ref = ((1.0 - alpha) * qpos_m[f0] + alpha * qpos_m[f1]).copy()
        qvel_ref = ((1.0 - alpha) * qvel_m[f0] + alpha * qvel_m[f1]).copy()

        # Apply subject-specific corrections (identical to generate_rsi_poses.py)
        if subject_settings is not None:
            hs  = subject_settings['height_scale']
            qpos_ref[subject_settings['ty_qpos_adr']]    *= hs
            qpos_ref[subject_settings['list_qpos_adr']]  += subject_settings['list_offset_rad']
            qpos_ref[subject_settings['ankle_r_qpos_adr']] += subject_settings['ankle_offset_rad']
            qvel_ref[subject_settings['ty_dof_adr']]     *= hs

        result = {
            'q_hat':    qpos_ref[joint_qpos_idx],
            'qdot_hat': qvel_ref[joint_qvel_idx],
        }

        # ── FK body references ────────────────────────────────────────────────
        if 'body_xpos' in data and 'body_vel' in data:
            bxpos_m = data['body_xpos']  # (T, 8, 3)
            bvel_m  = data['body_vel']   # (T, 8, 3)

            # Interpolate
            bxpos_ref = ((1.0 - alpha) * bxpos_m[f0] + alpha * bxpos_m[f1]).copy()
            bvel_ref  = ((1.0 - alpha) * bvel_m[f0]  + alpha * bvel_m[f1]).copy()

            # ── Treadmill offset ──────────────────────────────────────────────
            # The raw body_xpos FK was run on the .mot pelvis_tx which stays
            # nearly flat (treadmill trial: subject walks in place).  The true
            # overground-equivalent forward displacement is:
            #   X_global(t) = X_lab(t_start + elapsed) + motion_speed * elapsed
            # We apply only the *elapsed* contribution here so that at elapsed=0
            # bxpos_ref exactly matches the lab-frame position used to compute
            # the horizontal alignment offset in the environment.
            motion_speed = float(data.get('motion_speed', 0.0))
            bxpos_ref[:, 0] += motion_speed * elapsed  # accumulate forward (X)

            # Apply height scaling to Z-axis (index 2) of all body positions
            # This mirrors the pelvis_ty (which moves along Z in MuJoCo) correction.
            if subject_settings is not None:
                hs = subject_settings['height_scale']
                bxpos_ref[:, 2] *= hs
                bvel_ref[:, 2]  *= hs

            result['body_xpos_ref'] = bxpos_ref  # (8, 3)
            result['body_vel_ref']  = bvel_ref   # (8, 3)

        return result