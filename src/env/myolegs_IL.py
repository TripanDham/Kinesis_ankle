# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.

import os
import joblib
import numpy as np
from collections import OrderedDict, deque
from omegaconf import DictConfig, ListConfig
from typing import Dict, Iterator, Optional, Tuple
import scipy
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
from torch.optim import Adam

from pathlib import Path
import sys
path_root = Path(__file__).resolve().parents[2]
sys.path.append(str(path_root))

from src.env.myolegs_gail_task import MyoLegsGailTask
from src.utils.visual_capsule import add_visual_capsule
from src.utils.expert_ghost import ExpertGhost
from src.env.myolegs_gail_env import get_actuator_names
from src.KinesisCore.prostwalk_core import ProstWalkCore
from gail_airl_ppo.network import GAILDiscrim

import logging

logger = logging.getLogger(__name__)

class MyoLegsGAIL(MyoLegsGailTask):
    """
    MyoLegsRL focuses on GAIL training using OpenSim expert data.
    It returns a 32D joint-based observation (16 angles, 16 velocities) 
    with temporal history (s_t, s_t-1, s_t-2).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dtype = np.float32
        
        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)
        
        self.global_offset = np.zeros([1, 3])
        self.history_len = cfg.run.get("history_len", 6)
        self.history_buffer = deque(maxlen=self.history_len)

        super().__init__(cfg)
        
        self._setup_obs_mapping()
        self.setup_motionlib()
        
        # Discriminator receives same observations as actor (33D per frame)
        # NOTE: Must stay on CPU — compute_reward() is called in forked sampling workers
        # which cannot access GPU. The agent moves it to GPU for training only.
        obs_size = self.get_task_obs_size() 
        self.gail_disc = GAILDiscrim(
            state_shape=(obs_size,),
            action_shape=(0,), # State-only GAIL
            hidden_units=cfg.env.get("gail_hidden_units", (256, 256)),
            state_only=True
        )  # Stays on CPU
        
        self.optim_disc = Adam(self.gail_disc.parameters(), lr=cfg.learning.get("gail_lr", 1e-4))
        
        # Expert ghost visualization
        self.expert_ghost = ExpertGhost(self.mj_model, lateral_offset=-1.5)
        self.expert_motion_time = 0.0
        self.expert_motion_id = None

    def _setup_obs_mapping(self):
        """Pre-calculates qpos/qvel indices for the 30D observation vector."""
        # 1. Angles (13D)
        # Root (0D) - Pelvis tilt, list, rot removed
        root_angle_names = []
        # Right Leg (5): hip (3), knee (1), ankle (1)
        right_leg_names = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "osl_ankle_angle_r"]
        # Left Leg (5): hip (3), knee (1), ankle (1)
        left_leg_names = ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l"]
        
        angle_names = root_angle_names + right_leg_names + left_leg_names
        self.obs_qpos_idx = [self.mj_model.joint(n).qposadr[0] for n in angle_names]
        
        # 2. Velocities (16D)
        # Root (3): lin (3) - Angular (tilt, list, rotation) removed
        root_vel_names = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
        # Right/Left joint velocities
        vel_names = root_vel_names + right_leg_names + left_leg_names
        self.obs_qvel_idx = [self.mj_model.joint(n).dofadr[0] for n in vel_names]
        
        logger.info(f"Observation mapping initialized for 24D vector.")

    def setup_motionlib(self):
        """Initializes the motion library using ProstWalkCore."""
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self.motion_lib = ProstWalkCore(
            self.cfg.run, 
            joint_names=joint_names,
            mj_model=self.mj_model
        )
        self.motion_lib.load_motions(self.cfg.run)
        logger.info(f"Motion library initialized with {len(self.motion_lib.curr_motion_keys)} motions.")

    def get_disc_obs(self) -> np.ndarray:
        """
        Computes the 24D raw observation (10 angles + 13 velocities + 1 root height)
        matching the expert data format, excluding pelvis translation and target speed.
        """
        # 1. Angles (13D)
        angles = self.mj_data.qpos[self.obs_qpos_idx].astype(self.dtype)
        
        # 2. Velocities (16D)
        vels = self.mj_data.qvel[self.obs_qvel_idx].astype(self.dtype)

        self.curr_proprioception = angles # Used for height/upright rewards
        
        # 3. Root Height is removed from discriminator state
        return np.concatenate([angles, vels])

    def compute_task_obs(self) -> np.ndarray:
        """Returns the concatenated temporal history of disc observations."""
        raw_obs = self.get_disc_obs()
        self.history_buffer.append(raw_obs)
        
        # Pad if buffer is not full
        hist = list(self.history_buffer)
        while len(hist) < self.history_len:
            hist.insert(0, hist[0])
            
        return np.concatenate(hist)

    def get_task_obs_size(self) -> int:
        """Size of the GAIL history state (e.g. 23 * history_len)."""
        return 23 * self.history_len


    def get_gail_feature_names(self):
        """Returns the list of 23 feature names used in each frame of the GAIL state."""
        # 1. Angles (10D)
        right_leg_names = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "osl_ankle_angle_r"]
        left_leg_names = ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l"]
        angle_names = right_leg_names + left_leg_names
        
        # 2. Velocities (13D)
        root_vel_names = ["pelvis_tx_v", "pelvis_ty_v", "pelvis_tz_v"]
        vel_names = root_vel_names + [n + "_v" for n in angle_names]
        
        return angle_names + vel_names

    def init_myolegs(self):
        """
        Initializes the MyoLegs environment by loading the 'walk_right' keyframe,
        then overwrites the 10 tracked joint angles AND the 3 pelvis angles with 
        hardcoded expert values from tf01_0p6_01_rotated_ik.mot (Frame 0).
        All velocities are zeroed.
        """
        try:
            # 1. Load 'walk_right' keyframe for full qpos/qvel state baseline
            stand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, 'walk_right')
            if stand_id != -1:
                self.mj_data.qpos[:] = self.mj_model.key_qpos[stand_id]
                logger.info(f"Loaded 'walk_right' keyframe (id: {stand_id}) as baseline.")
            else:
                self.mj_data.qpos[:] = 0
                self.mj_data.qpos[1] = 0.88 # Fallback height
                logger.warning("Could not find 'walk_right' keyframe. Using zeroed qpos baseline.")

            # 2. Set all velocities to zero
            self.mj_data.qvel[:] = 0
            self.mj_data.qpos[1] = 0.89
            # 3. Overwrite only the 10 Tracked Joint Angles (Frame 0 of 0p4_01_rotated_ik.mot)
            expert_angles = [0.4650, -0.1353, 0.0303, 0.0529, 0.0815, 0.0175, 0.0639, -0.1207, 0.0127, 0.1504]
            # File: /media/tripan/Data/DDP/amputee_data/training_data_tf02_0p4/0p4_01_rotated_ik.mot
            # expert_angles = [
            #     0.4650, -0.1770, -0.1489, -0.8497, -0.0427, # Right Leg (Inverted knee/ankle)
            #     0.5475, -0.0114, -0.1414, -0.2828, -0.1639  # Left Leg (Inverted knee/ankle)
            # ]
            
            for i, idx in enumerate(self.obs_qpos_idx):
                self.mj_data.qpos[idx] = expert_angles[i]
            
            # 4. Overwrite Pelvis Angles (Frame 0 of 0p4_01_rotated_ik.mot)
            # Degrees: tilt: -20.5815, list: 2.0785, rotation: -7.6137
            # tf01_0p6_01
            self.mj_data.qpos[3] = -0.1177
            self.mj_data.qpos[4] = 0.0512 
            self.mj_data.qpos[5] = -0.0129 
            # tf02_0p4_01
            # self.mj_data.qpos[3] = -0.3592 
            # self.mj_data.qpos[4] = 0.0363  
            # self.mj_data.qpos[5] = -0.1329 

            logger.info("Initialized with walk_right baseline, expert (tf01_0p6_01) joints + pelvis angles, and zeroed velocities.")

        except Exception as e:
            logger.warning(f"Error during init_myolegs: {e}")
            self.mj_data.qpos[:] = 0
            self.mj_data.qvel[:] = 0
            self.mj_data.qpos[1] = 0.88
            
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

    def reset_task(self, options=None):
        """Resets task-specific state."""
        self.history_buffer.clear()
        
        # Pick a target speed from the expert library
        if options is not None and "target_speed" in options:
            self.target_speed = options["target_speed"]
        elif self.cfg.run.test and getattr(self.cfg.run, "eval_target_speed", None) is not None:
            self.target_speed = self.cfg.run.eval_target_speed
        else:
            self.target_speed = np.random.choice(self.motion_lib.available_speeds)
            
        self.biomechanics_data = []
        
        # Pick an expert motion for the ghost visualization
        if hasattr(self, 'motion_lib') and len(self.motion_lib.available_speeds) > 0:
            # Try to get a motion matching target speed
            try:
                motion_ids = self.motion_lib.sample_motions_by_velocity(self.target_speed, n=1)
                self.expert_motion_id = motion_ids[0]
                self.expert_motion_time = 0.0
            except Exception:
                self.expert_motion_id = 0
                self.expert_motion_time = 0.0
        
        logger.info(f"Target speed for this episode: {self.target_speed}")

    # compute_task_obs is defined above and dynamically returns the history

    def draw_task(self):
        """Draws expert ghost model in the viewer each render frame."""
        if self.headless or self.viewer is None:
            return
        if not self.expert_ghost.enabled:
            # Clear any leftover ghost geoms when disabled
            with self.viewer.lock():
                self.viewer._user_scn.ngeom = 0
            return
        
        # Get current expert reference qpos from the motion library
        try:
            if self.expert_motion_id is not None:
                motion_state = self.motion_lib.get_motion_state_intervaled(
                    np.array([self.expert_motion_id]),
                    np.array([self.expert_motion_time])
                )
                expert_qpos = motion_state['qpos'][0]
                self.expert_ghost.update_pose(expert_qpos)
                
                # Offset ghost to be beside the agent
                agent_root = self.mj_data.qpos[0:3].copy()
                self.expert_ghost.apply_offset(agent_root)
                
                # Render ghost into user scene
                with self.viewer.lock():
                    self.expert_ghost.draw(self.viewer)
                
                # Advance expert motion time
                self.expert_motion_time += self.dt
                motion_len = self.motion_lib.get_motion_length(np.array([self.expert_motion_id]))[0]
                if self.expert_motion_time > motion_len:
                    self.expert_motion_time = 0.0
        except Exception as e:
            pass  # Silently skip if motion state is unavailable

    def create_task_visualization(self):
        pass

    def set_normalizer(self, normalizer):
        """Sets the normalizer reference to allow reward-time state checking."""
        self.normalizer = normalizer

    def compute_reward(self, action: Optional[np.ndarray] = None) -> float:
        """GAIL Reward using the Discriminator + Velocity Matching."""
        gail_obs = self.compute_task_obs()
        device = next(self.gail_disc.parameters()).device
        obs_tensor = torch.as_tensor(gail_obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = torch.zeros((1, 0), device=device)
        
        with torch.no_grad():
            # calculate_reward returns -log(1-D) which is in [0, inf)
            im_reward = self.gail_disc.calculate_reward(obs_tensor, a_tensor).item()
            
            # STABILIZATION: Clamp the imitation reward to prevent explosion
            # A reward of 2.0 corresponds to D=0.86, which is a strong signal but not destabilizing.
            im_reward = np.clip(im_reward, 0.0, 2.0)
            
        dist_reward = self.compute_distance_reward()
        upright_reward = self.compute_upright_reward()
        
        # Split Energy Reward
        muscle_effort = self.compute_muscle_effort(action)
        motor_effort = self.compute_motor_effort(action)
        
        # Ankle Delta Penalty (for multi-rate impedance)
        w_ankle_delta = self.cfg.env.reward_specs.get("w_ankle_delta", 0.01)
        ankle_delta_penalty = np.sum(np.square(self.delta_ankle_action))
        
        w_muscle = self.cfg.env.reward_specs.get("w_energy", 0.01)
        w_motor = self.cfg.env.reward_specs.get("w_motor_effort", 0.1)

        # # Simple Ankle Out-of-Bounds Penalty (-5 if > +/- 20 degrees)
        # j_ankle = self.mj_model.joint("osl_ankle_angle_r").id
        # q_ankle = self.mj_data.qpos[self.mj_model.jnt_qposadr[j_ankle]]
        # limit_rad = 20 * np.pi / 180.0
        # ankle_limit_penalty = 5.0 if np.abs(q_ankle) > limit_rad else 0.0

        # NEW: State-wide Out-of-Bounds Penalty (based on normalizer)
        state_oob_penalty = 0.0
        if hasattr(self, 'normalizer') and self.normalizer is not None and self.normalizer.n > 1000:
            # We use the normalizer's clip value (usually 5.0) as the "out-of-bounds" threshold
            norm_clip = getattr(self.normalizer, 'clip', 5.0)
            with torch.no_grad():
                # Normalize the current task observation (slice normalizer to match gail_obs)
                gail_obs_size = len(gail_obs)
                mean = self.normalizer.mean[:gail_obs_size].cpu().numpy()
                std = self.normalizer.std[:gail_obs_size].cpu().numpy()
                normalized_obs = (gail_obs - mean) / (std + 1e-8)
                
                # Penalize only the magnitude exceeding the clip threshold
                excess = np.maximum(0, np.abs(normalized_obs) - norm_clip)
                state_oob_penalty = np.sum(np.square(excess))
        
        w_state_oob = self.cfg.env.reward_specs.get("w_state_oob", 0.1)

        # DELAYED GAIL START: Ignore imitation reward before 3000 epochs
        current_epoch = getattr(self, 'current_epoch', 0)
        im_weight = 1.0 if current_epoch >= 3000 else 0.0

        reward = (im_weight * im_reward + 
                  0.2 * dist_reward + 
                  0.3 * upright_reward - 
                  w_muscle * muscle_effort - 
                  w_motor * motor_effort -
                  w_ankle_delta * ankle_delta_penalty -
                  w_state_oob * state_oob_penalty)
        
        self.reward_info = {
            "imitation_reward_gail": im_reward, 
            "distance_reward": dist_reward,
            "upright_reward": upright_reward,
            "muscle_effort": muscle_effort,
            "motor_effort": motor_effort,
            "ankle_delta_penalty": ankle_delta_penalty,
            "state_oob_penalty": state_oob_penalty,
            "total_reward": reward
        }
        return reward

    def compute_distance_reward(self) -> float:
        """Rewards the absolute forward distance moved by the pelvis."""
        if getattr(self, 'cur_t', 0) <= 1 or not hasattr(self, 'start_pelvis_x'):
            self.start_pelvis_x = self.mj_data.qpos[0]
            
        distance = self.mj_data.qpos[0] - self.start_pelvis_x
        return float(distance)

    def compute_muscle_effort(self, action: np.ndarray) -> float:
        """Computes effort penalty for biological muscles."""
        if action is None: return 0.0
        muscle_acts = action[self.muscle_idx]
        return np.sum(np.square(muscle_acts))

    def compute_motor_effort(self, action: np.ndarray) -> float:
        """Computes effort penalty for prosthetic motors (impedance params or direct torque)."""
        if action is None: return 0.0
        motor_acts = action[self.motor_idx]
        return np.sum(np.square(motor_acts))

    def compute_upright_reward(self) -> float:
        """
        Computes the reward for maintaining an upright posture.

        The reward is based on the angles of tilt in the forward and sideways directions, 
        calculated using trigonometric components of the root tilt.

        Returns:
            float: The upright reward, where a value close to 1 indicates a nearly upright posture.
        """
        root_rot_euler = self.curr_proprioception[0:3]
        upright_trigs = np.array([np.cos(root_rot_euler[0]), np.sin(root_rot_euler[0]), np.cos(root_rot_euler[1]), np.sin(root_rot_euler[1])])
        fall_forward = np.angle(upright_trigs[0] + 1j * upright_trigs[1])
        fall_sideways = np.angle(upright_trigs[2] + 1j * upright_trigs[3])
        upright_reward = np.exp(-3 * (fall_forward ** 2 + fall_sideways ** 2))
        return upright_reward

    def compute_reset(self) -> Tuple[bool, bool]:
        """Basic stability and time-based reset."""
        # Y-up model: pelvis_ty (index 1) is the height
        fell = self.mj_data.qpos[1] < 0.5 or self.mj_data.qpos[1] > 1.2
        truncated = self.cur_t >= self.max_episode_length
        return fell, truncated

    def initialize_env_params(self, cfg: DictConfig) -> None:
        self.max_episode_length = cfg.env.get("max_episode_length", 300)
        self.muscle_condition = cfg.env.get("muscle_condition", "")

    def initialize_run_params(self, cfg: DictConfig) -> None:
        self.motion_start_idx = cfg.run.motion_id
        self.num_motion_max = cfg.run.num_motions
        self.motion_file = cfg.run.motion_file
        self.initial_pose_file = cfg.run.initial_pose_file
        self.device = cfg.run.get("device", "cpu")
        self.num_threads = cfg.run.get("num_threads", 1)

    def record_biomechanics(self):
        """Records biomechanics state at the current timestep testing."""
        if not self.cfg.run.test or not getattr(self.cfg.run, "record_biomechanics", False):
            return
            
        # Record heights for ground clearance diagnostics
        # osl_foot_assembly (12), calcn_l (16)
        # We use xpos[index, 2] for the Z-coordinate
        right_foot_z = self.mj_data.xpos[12, 2]
        left_foot_z = self.mj_data.xpos[16, 2]
        
        # Get ankle heights as well (joints 9 and 18)
        # Note: bodies associated with these joints might be better. 
        # For osl_ankle, it's body 'osl_ankle_assembly' (11).
        # For ankle_l, it's body 'talus_l' (15).
        right_ankle_z = self.mj_data.xpos[11, 2]
        left_ankle_z = self.mj_data.xpos[15, 2]

        # Calculate Net Torque (aggregate joint moments) as requested:
        # qfrc_actuator + qfrc_applied + qfrc_passive - qfrc_bias
        # We use getattr for robustness across MuJoCo versions
        q_act = self.mj_data.qfrc_actuator.copy()
        q_app = getattr(self.mj_data, "qfrc_applied", np.zeros_like(q_act)).copy()
        q_pas = getattr(self.mj_data, "qfrc_passive", np.zeros_like(q_act)).copy()
        q_bia = getattr(self.mj_data, "qfrc_bias", np.zeros_like(q_act)).copy()
        
        net_torque = q_act + q_app + q_pas - q_bia

        data = {
            "qpos": self.mj_data.qpos.copy(),
            "qvel": self.mj_data.qvel.copy(),
            "ctrl": self.mj_data.ctrl.copy(),
            "qfrc_actuator": net_torque, # Rename to qfrc_actuator for plotter compatibility or use new key
            "qfrc_actuator_only": self.mj_data.qfrc_actuator.copy(),
            "actuator_force": self.mj_data.actuator_force.copy(),
            "actuator_activation": getattr(self.mj_data, "actuator_activation", getattr(self.mj_data, "act", np.zeros(0))).copy(),
            "impedance": getattr(self, "last_impedance", {}).copy(),
            "heights": {
                "right_foot": right_foot_z,
                "left_foot": left_foot_z,
                "right_ankle": right_ankle_z,
                "left_ankle": left_ankle_z
            }
        }
        self.biomechanics_data.append(data)

    def post_physics_step(self, action):
        """Overrides base post_physics_step to include biomechanics tracking."""
        obs, reward, terminated, truncated, info = super().post_physics_step(action)
        self.record_biomechanics()
        
        if terminated or truncated:
            if getattr(self.cfg.run, "record_biomechanics", False):
                info["biomechanics_data"] = self.biomechanics_data
                
        return obs, reward, terminated, truncated, info