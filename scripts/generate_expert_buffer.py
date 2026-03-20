import os
import torch
import numpy as np
import sys
sys.path.append(os.getcwd())
from gail_airl_ppo.buffer import SerializedBuffer
from src.env.myolegs_IL import MyoLegsGAIL
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import mujoco
import hydra
from hydra import compose, initialize

def generate_expert_buffer(motion_file, output_path, history_len=3, device="cpu"):
    """
    Loads expert motions and creates a SerializedBuffer by running them through the environment.
    """
    # Initialize and compose Hydra config
    # version_base=None or "1.1" to avoid warning
    with initialize(config_path="../cfg", version_base=None):
        cfg = compose(config_name="config", overrides=[
            f"env=myolegs_gail",
            f"run.motion_file={motion_file}",
            f"run.num_motions=1000",
            f"run.headless=True",
            f"run.history_len={history_len}"
        ])
    
    # Initialize environment
    env = MyoLegsGAIL(cfg)
    
    states = []
    
    # Use actual loaded motions from library
    num_motions = len(env.motion_lib.curr_motion_keys)
    print(f"Processing {num_motions} motions...")
    
    for m_id in range(num_motions):
        # Reset environment (now only for physical state reset)
        env.reset()
        
        motion_len = env.motion_lib.get_motion_length(m_id)
        fps = env.motion_lib._motion_fps[m_id]
        num_frames = int(motion_len * fps)
        
        print(f"Motion {m_id}: {num_frames} frames")
        
        # Clear/Init history in the environment
        env.history_buffer.clear()
        
        for f in range(num_frames):
            # Get expert state for this frame directly from motion_lib
            sim_time = f / fps
            ref_dict = env.motion_lib.get_motion_state_intervaled(
                np.array([m_id]), np.array([sim_time]), env.global_offset
            )
            
            # Set simulation state to expert state
            env.mj_data.qpos[:] = ref_dict.qpos[0]
            env.mj_data.qvel[:] = ref_dict.qvel[0]
            mujoco.mj_forward(env.mj_model, env.mj_data)
            
            # compute_observations() handles:
            # 1. get_disc_obs() (30D)
            # 2. updating history_buffer
            # 3. padding and concatenating to history_len * 30D
            hist_obs = env.compute_observations()
            states.append(hist_obs)

    states = np.array(states, dtype=np.float32)
    print(f"Collected total buffer size: {states.shape}")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save({
        'state': torch.from_numpy(states),
    }, output_path)
    
    print(f"Expert buffer saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, default="/media/tripan/Data/DDP/amputee_data/training_data")
    parser.add_argument("--output_path", type=str, default="data/buffers/expert_buffer.pth")
    parser.add_argument("--history_len", type=int, default=6)
    args = parser.parse_args()
    
    generate_expert_buffer(args.motion_file, args.output_path, args.history_len)
