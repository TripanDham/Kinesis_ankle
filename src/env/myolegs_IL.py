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
    
    # Hardcoded Expert Distribution (23D) for Out-of-Bounds Penalty
    EXPERT_MEAN = np.array([
        0.134134, -0.117375, -0.083615, -0.179425, -0.155010, 0.189645, 0.039989, -0.142105, -0.308973, -0.104356, 
        0.598441, 0.002280, 0.002313, -0.041557, 0.009529, -0.012929, 0.022744, 0.001598, 0.003744, -0.011868, 
        0.005017, -0.015650, 0.007822
    ])

    EXPERT_STD = np.array([
        0.245877, 0.030119, 0.081604, 0.272526, 0.097627, 0.210640, 0.058692, 0.063381, 0.363935, 0.111683, 
        0.120364, 0.132024, 0.105852, 1.508349, 0.380830, 0.558943, 2.275996, 0.693375, 1.217393, 0.542282, 
        0.708819, 2.892466, 1.253362
    ])

    def __init__(self, cfg):
        self.cfg = cfg
        self.dtype = np.float32
        
        self.obs_tracking_reference = self.cfg.env.get("obs_tracking_reference", True)
        
        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)
        
        self.global_offset = np.zeros([1, 3])
        self.history_len = cfg.run.get("history_len", 6)
        self.history_buffer = deque(maxlen=self.history_len)

        super().__init__(cfg)
        
        self._setup_obs_mapping()
        self.setup_motionlib()
        self._load_rsi_poses()
        
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
        
        # WGAN-GP reward normalization (EMA)
        self._reward_ema_alpha = cfg.learning.get("wgan_reward_ema", 0.9999)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
        
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

    def _load_rsi_poses(self):
        """Loads pre-computed RSI poses and tracking metadata from .npz file."""
        rsi_path = self.cfg.run.get("rsi_poses_path", None)
        self.rsi_init_vel = self.cfg.run.get("rsi_init_vel", True)

        if rsi_path and os.path.exists(rsi_path):
            data = np.load(rsi_path, allow_pickle=True)
            self.rsi_qpos = data['rsi_qpos']   # (N, nq)
            self.rsi_qvel = data['rsi_qvel']   # (N, nv)
            # Tracking metadata (added by updated generate_rsi_poses.py)
            self.rsi_frame_idx  = data['rsi_frame_idx']   # (N,) int
            self.rsi_fps        = data['rsi_fps']          # (N,) float
            self.rsi_motion_key = data['rsi_motion_key']  # (N,) str
            self.rsi_subject_id = data['rsi_subject_id']  # (N,) str
            self.rsi_enabled = True
            logger.info(f"RSI loaded: {self.rsi_qpos.shape[0]} poses from {rsi_path} "
                        f"(vel_init={'ON' if self.rsi_init_vel else 'OFF'})")
        else:
            self.rsi_enabled = False
            self.rsi_frame_idx  = None
            self.rsi_fps        = None
            self.rsi_motion_key = None
            self.rsi_subject_id = None
            if rsi_path:
                logger.warning(f"RSI file not found: {rsi_path}. Falling back to keyframe init.")
            else:
                logger.info("RSI disabled (no rsi_poses_path configured).")

        # Build subject correction lookup (MuJoCo address lookups done once here).
        # Replicates the exact corrections applied in generate_rsi_poses.py.
        from scripts.generate_rsi_poses import SUBJECT_SETTINGS
        self._subject_correction_map = {}
        for subj, settings in SUBJECT_SETTINGS.items():
            self._subject_correction_map[subj] = {
                'height_scale':     settings['height_scale'],
                'list_offset_rad':  np.deg2rad(settings['pelvis_list_offset_deg']),
                'ankle_offset_rad': np.deg2rad(settings['ankle_r_offset_deg']),
                'ty_qpos_adr':      self.mj_model.joint('pelvis_ty').qposadr[0],
                'list_qpos_adr':    self.mj_model.joint('pelvis_list').qposadr[0],
                'ankle_r_qpos_adr': self.mj_model.joint('osl_ankle_angle_r').qposadr[0],
                'ty_dof_adr':       self.mj_model.joint('pelvis_ty').dofadr[0],
            }

        # Tracking config — inherited from yaml
        self._sim_dt            = self.mj_model.opt.timestep * getattr(self, 'control_freq_inv', 1)
        self._tracking_interval = self.cfg.env.get('tracking_interval', 0.1)
        self.w_track_pos        = self.cfg.env.get('w_track_pos', 2.0)
        self.w_track_vel        = self.cfg.env.get('w_track_vel', 0.1)
        self.w_track            = self.cfg.env.get('tracking_reward_weight', 1.0)

        # Tracking runtime state (reset each episode in init_myolegs)
        self._tracking_motion_key    = None
        self._tracking_t_start       = 0.0
        self._tracking_subject       = None
        self._tracking_elapsed       = 0.0
        self._tracking_last_ref_time = -np.inf
        
        # Initialise reference states as zeros for consistent observation sizing
        num_joints = len(self.obs_qpos_idx)
        self._q_hat                  = np.zeros(num_joints, dtype=self.dtype)
        self._qdot_hat               = np.zeros(num_joints, dtype=self.dtype)
        
        # joint-only velocity indices — obs_qvel_idx[3:] drops the 3 pelvis linear vels
        self._track_qvel_idx         = self.obs_qvel_idx[3:]


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

    def compute_proprioception(self) -> np.ndarray:
        """
        Overrides base proprioception to optionally include tracking references.
        """
        # Lazy initialization for early calls during base class __init__
        if not hasattr(self, "obs_qpos_idx"):
            self._setup_obs_mapping()
            num_joints = len(self.obs_qpos_idx)
            self._q_hat = np.zeros(num_joints, dtype=self.dtype)
            self._qdot_hat = np.zeros(num_joints, dtype=self.dtype)
            self._track_qvel_idx = self.obs_qvel_idx[3:]

        # Call base implementation to get standard features (target_speed, activations, contacts)
        from src.env.myolegs_gail_env import MyoLegsGailEnv
        prop_array = MyoLegsGailEnv.compute_proprioception(self)
        
        if self.obs_tracking_reference:
            # Append references to the end of proprioception
            # Update self.proprioception dict to include these for get_self_obs_size() consistency
            self.proprioception["q_hat"] = self._q_hat
            self.proprioception["qdot_hat"] = self._qdot_hat
            return np.concatenate([prop_array, self._q_hat, self._qdot_hat])
        
        return prop_array

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
        Initializes the MyoLegs environment using Reference State Initialization (RSI).
        
        If RSI is enabled, randomly samples a pre-computed (qpos, qvel) pair from
        the RSI pose pool. These poses were generated from expert .mot files with
        subject-specific height scaling, ankle offsets, and pelvis tilt corrections.
        
        Falls back to the 'stand' keyframe if RSI is not available.
        """
        if self.rsi_enabled:
            # Sample a random pose from the RSI pool
            idx = np.random.randint(0, len(self.rsi_qpos))
            
            self.mj_data.qpos[:] = self.rsi_qpos[idx]
            
            if self.rsi_init_vel:
                self.mj_data.qvel[:] = self.rsi_qvel[idx]
            else:
                self.mj_data.qvel[:] = 0.0

            # --- Sync tracking state with this RSI pose ---
            if self.rsi_motion_key is not None:
                self._tracking_motion_key    = str(self.rsi_motion_key[idx])
                self._tracking_t_start       = float(self.rsi_frame_idx[idx]) / float(self.rsi_fps[idx])
                subj_id                      = str(self.rsi_subject_id[idx])
                self._tracking_subject       = self._subject_correction_map.get(subj_id)
            else:
                self._tracking_motion_key = None

            # Reset tracking clock
            self._tracking_elapsed       = 0.0
            self._tracking_last_ref_time = -np.inf
            
            # Reset references to zeros (never None, for observation consistency)
            num_joints = len(self.obs_qpos_idx)
            self._q_hat = np.zeros(num_joints, dtype=self.dtype)
            self._qdot_hat = np.zeros(num_joints, dtype=self.dtype)
        else:
            # Fallback: use 'stand' keyframe
            stand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
            if stand_id != -1:
                self.mj_data.qpos[:] = self.mj_model.key_qpos[stand_id]
                self.mj_data.qvel[:] = 0.0
            else:
                self.mj_data.qpos[:] = 0
                self.mj_data.qvel[:] = 0
                self.mj_data.qpos[1] = 0.91
            
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
        """WGAN-GP Reward using the Critic + Velocity Matching."""
        gail_obs = self.compute_task_obs()
        device = next(self.gail_disc.parameters()).device
        obs_tensor = torch.as_tensor(gail_obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = torch.zeros((1, 0), device=device)
        
        with torch.no_grad():
            # WGAN reward: raw critic output (higher = more expert-like)
            im_reward_raw = self.gail_disc.calculate_reward_wgan(obs_tensor, a_tensor).item()
            
            # EMA normalization to [0, 1]
            self._reward_count += 1
            alpha = self._reward_ema_alpha
            if self._reward_count == 1:
                self._reward_mean = im_reward_raw
                self._reward_var = 1.0
            else:
                self._reward_mean = alpha * self._reward_mean + (1 - alpha) * im_reward_raw
                self._reward_var = alpha * self._reward_var + (1 - alpha) * (im_reward_raw - self._reward_mean) ** 2
            
            reward_std = np.sqrt(self._reward_var) + 1e-8
            im_reward = np.clip((im_reward_raw - self._reward_mean) / reward_std, -1.0, 1.0)
            im_reward = (im_reward + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]

        # Advance tracking clock and fetch reference if interval has elapsed
        self._tracking_elapsed += self._sim_dt
        if (self._tracking_motion_key is not None and
                self._tracking_elapsed - self._tracking_last_ref_time >= self._tracking_interval):
            try:
                ref = self.motion_lib.get_reference_state(
                    motion_key       = self._tracking_motion_key,
                    t_start          = self._tracking_t_start,
                    elapsed          = self._tracking_elapsed,
                    joint_qpos_idx   = self.obs_qpos_idx,
                    joint_qvel_idx   = self._track_qvel_idx,
                    subject_settings = self._tracking_subject,
                )
                self._q_hat    = ref['q_hat']
                self._qdot_hat = ref['qdot_hat']
            except KeyError:
                pass  # motion_key not yet loaded — silently skip
            self._tracking_last_ref_time = self._tracking_elapsed
            
        vel_reward = self.compute_velocity_reward()
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

        # NEW: State-wide Out-of-Bounds Penalty (hardcoded expert stats)
        state_oob_penalty = 0.0
        # We use a fixed 5.0 std threshold for "out-of-bounds"
        norm_clip = 5.0
        
        # Normalize the current task observation (23D)
        normalized_obs = (gail_obs - self.EXPERT_MEAN) / (self.EXPERT_STD + 1e-8)
        
        # Penalize based on the number of features outside the threshold
        state_oob_penalty = np.sum(np.abs(normalized_obs) > norm_clip)
        
        w_state_oob = self.cfg.env.reward_specs.get("w_state_oob", 0.1)

        tracking_reward = self.compute_tracking_reward()

        reward = (im_reward + 
                  0.2 * vel_reward + 
                  0.3 * upright_reward +
                  self.w_track * tracking_reward -
                  0.01 * muscle_effort - 
                  0.05 * motor_effort -
                  0 * ankle_delta_penalty -
                  0.01 * state_oob_penalty)
        
        self.reward_info = {
            "imitation_reward_gail": im_reward, 
            "velocity_reward": vel_reward,
            "upright_reward": upright_reward,
            "tracking_reward": tracking_reward,
            "muscle_effort": muscle_effort,
            "motor_effort": motor_effort,
            "ankle_delta_penalty": ankle_delta_penalty,
            "state_oob_penalty": state_oob_penalty,
            "total_reward": reward
        }
        return reward

    def compute_velocity_reward(self) -> float:
        """Rewards forward pelvis velocity matching the treadmill target (0.6 m/s).

        R_vel = exp( -2.0 * (v_x_pelvis - 0.6)^2 )

        pelvis_tx is dofadr[0] in the hinge-root model.
        """
        pelvis_tx_dof = self.mj_model.joint('pelvis_tx').dofadr[0]
        vx = float(self.mj_data.qvel[pelvis_tx_dof])
        return float(np.exp(-2.0 * (vx - 0.6) ** 2))

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

    def compute_tracking_reward(self) -> float:
        """
        Computes R_track = exp(-w1*||q - q_hat||^2 - w2*||qdot - qdot_hat||^2)
        for joints only (no pelvis translation/rotation).
        Returns 0.0 if the reference has not been fetched yet.
        """
        if self._q_hat is None or self._qdot_hat is None:
            return 0.0
        q    = self.mj_data.qpos[self.obs_qpos_idx]    # joint angles (10D)
        qdot = self.mj_data.qvel[self._track_qvel_idx] # joint vels (10D, no pelvis)
        dq   = q    - self._q_hat
        dqd  = qdot - self._qdot_hat
        return float(np.exp(
            -self.w_track_pos * float(dq  @ dq) 
            -self.w_track_vel * float(dqd @ dqd)
        ))

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