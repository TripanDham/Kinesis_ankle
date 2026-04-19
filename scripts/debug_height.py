import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import mujoco

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.myolegs_IL import MyoLegsGAIL
from src.agents.agent_gail import AgentGAIL

@hydra.main(config_path="../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg.run.num_threads = 1
    
    print("Initialising Agent & Env...")
    agent = AgentGAIL(
        cfg=cfg, 
        dtype=torch.float32,
        device="cpu",
        training=True
    )
    env = agent.env
    
    print("\n--- Initial State (Before Reset) ---")
    pelvis_idx = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"Pelvis Body ID: {pelvis_idx}")
    print(f"Pelvis xpos: {env.mj_data.xpos[pelvis_idx]}")
    
    print("\n--- Resetting Environment ---")
    obs, _ = env.reset()
    
    print(f"Observation Size: {len(obs)}")
    print(f"Task Obs Size: {env.get_task_obs_size()}")
    
    proprio_part = obs[180:]
    print(f"Full Proprioception Vector ({len(proprio_part)}D): {proprio_part}")
    
    print(f"\nInternal MyoLegsGAIL.proprioception dict keys: {env.proprioception.keys()}")
    for k, v in env.proprioception.items():
        print(f"  {k}: {v}")
        
    print(f"\nPelvis xpos after reset: {env.mj_data.xpos[pelvis_idx]}")
    print(f"Pelvis qpos (root): {env.mj_data.qpos[:7]}")

if __name__ == "__main__":
    main()
