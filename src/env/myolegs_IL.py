# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import os
import sys
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple

sys.path.append(os.getcwd())

from src.env.myolegs_task import MyoLegsTask
from scipy.spatial.transform import Rotation as sRot

class MyoLegsIL(MyoLegsTask):
    """
    Imitation Learning environment for MyoLegs.
    Loads expert motion data from .mot files and implements a discriminator-based reward.
    """
    def __init__(self, cfg):
        self.expert_data_path = "/media/tripan/Data/DDP/amputee_data/10308443/TF01/Vicon Workspace/tf01_0p6.mot"
        self.ref_motion = self.load_mot_data(self.expert_data_path)
        super().__init__(cfg=cfg)

    def load_mot_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Parses an OpenSim .mot file.
        Returns a dictionary mapping column names to numpy arrays.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find the end of the header
        header_end = 0
        in_degrees = False
        for i, line in enumerate(lines):
            if 'inDegrees=yes' in line:
                in_degrees = True
            if 'endheader' in line:
                header_end = i + 1
                break
        
        # Column names
        columns = lines[header_end].strip().split()
        data_lines = lines[header_end + 1:]
        
        data = np.array([line.strip().split() for line in data_lines if line.strip()], dtype=np.float32)
        
        motion_dict = {}
        for i, col in enumerate(columns):
            if in_degrees and col != 'time' and not col.endswith('_t'): # pelvis_tx/ty/tz are in meters
                # Check if it's a rotation (usually ends in _angle or starts with pelvis_tilt/list/rotation)
                # For safety, convert everything that isn't time or tx/ty/tz if inDegrees is yes
                if 'pelvis_t' not in col:
                     motion_dict[col] = np.deg2rad(data[:, i])
                else:
                     motion_dict[col] = data[:, i]
            else:
                motion_dict[col] = data[:, i]
        
        return motion_dict

    def get_task_obs_size(self) -> int:
        """Returns the size of the task-specific observations."""
        return 0 # No task-specific obs for now

    def reset_task(self, options=None):
        """Resets task-specific state."""
        pass

    def compute_task_obs(self) -> np.ndarray:
        """Computes task-specific observations."""
        return np.array([], dtype=self.dtype)

    def draw_task(self):
        """Draws task-specific visualization."""
        pass

    def create_task_visualization(self):
        """Creates task-specific visualization."""
        pass

    def compute_reward(self, action: np.ndarray) -> float:
        """
        Implements reward structure: r_upright + r_actions + r_com + r_discriminator
        """
        # Placeholder for discriminator reward
        r_dist = 0.0 # From GAIL discriminator
        
        # Upright reward (from root orientation)
        # In myolegs.xml, torso is part of the root body.
        # Fixed torso is rigid relative to pelvis? No, body 'torso' is child of 'root'.
        # Actually, let's look at root orientation.
        root_quat = self.mj_data.qpos[3:7] # Mujoco wxyz
        # Upright means quat is close to [1, 0, 0, 0] if aligned? 
        # Actually, let's use the upright reward from MyoLegs base if available.
        # BaseEnv has compute_proprioception which calculates root_tilt.
        r_upright = np.exp(-10.0 * (1.0 - root_quat[0]**2)) 

        # Action penalty
        r_actions = -0.01 * np.sum(np.square(action))

        # COM velocity reward (target speed)
        target_vel = self.cfg.env.get("target_speed", 0.6)
        # Root linear velocity is in sensordata if sensors are defined, 
        # or we can get it from mj_data.qvel[:3]
        current_vel = self.mj_data.qvel[0] # Root x-velocity
        r_com = np.exp(-2.0 * (current_vel - target_vel)**2)

        reward = r_upright + r_actions + r_com + r_dist
        
        self.reward_info = {
            "r_upright": r_upright,
            "r_actions": r_actions,
            "r_com": r_com,
            "r_dist": r_dist,
            "total_reward": reward
        }
        
        return reward

    def reset_myolegs(self):
        """
        Initializes agent pose from expert data.
        """
        # Initial pose from frame 0
        self.mj_data.qpos[:] = 0
        self.mj_data.qvel[:] = 0
        
        # Pelvis position (In .mot: pelvis_tx, pelvis_ty, pelvis_tz)
        # OpenSim: Y is up. MuJoCo: Z is up.
        # Mapping: MuJoCo X = OpenSim X, MuJoCo Y = OpenSim Z, MuJoCo Z = OpenSim Y
        self.mj_data.qpos[0] = self.ref_motion['pelvis_tx'][0]
        self.mj_data.qpos[1] = self.ref_motion['pelvis_tz'][0]
        self.mj_data.qpos[2] = self.ref_motion['pelvis_ty'][0]
        
        # Pelvis rotation: .mot has tilt, list, rotation (XYZ eulers in degrees)
        # This needs careful mapping to MuJoCo root quat.
        # For now, keep it identity or zero.
        
        # Joint angles
        mapping = {
            "hip_flexion_r": "hip_flexion_r",
            "hip_adduction_r": "hip_adduction_r",
            "hip_rotation_r": "hip_rotation_r",
            "knee_angle_r": "knee_angle_r",
            "ankle_angle_r": "ankle_angle_r",
            "subtalar_angle_r": "subtalar_angle_r",
            "mtp_angle_r": "mtp_angle_r",
            "hip_flexion_l": "hip_flexion_l",
            "hip_adduction_l": "hip_adduction_l",
            "hip_rotation_l": "hip_rotation_l",
            "knee_angle_l": "knee_angle_l",
            "ankle_angle_l": "ankle_angle_l",
            "subtalar_angle_l": "subtalar_angle_l",
            "mtp_angle_l": "mtp_angle_l",
        }
        
        for mot_joint, sim_joint in mapping.items():
            if mot_joint in self.ref_motion:
                try:
                    jnt_id = self.mj_model.joint(sim_joint).id
                    qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
                    self.mj_data.qpos[qpos_adr] = self.ref_motion[mot_joint][0]
                except Exception:
                    pass

        self.forward_sim()
