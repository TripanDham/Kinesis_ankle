import hydra
from omegaconf import DictConfig
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from src.env.myolegs_IL import MyoLegsGAIL
import mujoco

@hydra.main(config_path="../cfg", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    env = MyoLegsGAIL(cfg)
    obs = env.reset()
    
    # Take a step to compute the tracking references
    action = np.zeros(env.action_space.shape)
    _, reward, _, _, info = env.step(action)
    
    ref_xpos = env._body_pos_hat.reshape(-1, 3).copy()
    sim_xpos = env.mj_data.xpos[env._track_body_ids].copy()
    
    from src.KinesisCore.prostwalk_core import TRACKED_BODY_NAMES
    
    print("\nInitial Position Errors AFTER step:")
    for i, name in enumerate(TRACKED_BODY_NAMES):
        diff = sim_xpos[i] - ref_xpos[i]
        err = np.sum(diff**2)
        print(f"  {name:10s}: \n    sim = {sim_xpos[i]}\n    ref = {ref_xpos[i]}\n    diff = {diff}")

    print("\nPos reward:", info['body_pos_reward'])
    print("Vel reward:", info['body_vel_reward'])

if __name__ == "__main__":
    main()
