import hydra
from omegaconf import DictConfig
import sys
import os
sys.path.append(os.getcwd())

from src.env.myolegs_IL import MyoLegsGAIL

@hydra.main(config_path="../cfg", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    env = MyoLegsGAIL(cfg)
    obs = env.reset()
    print("Env reset successful!")
    print("Proprioception size:", env.get_self_obs_size())
    print("Task obs size:", env.get_task_obs_size())
    print("Total obs size:", len(obs[0]) if isinstance(obs, tuple) else len(obs))
    
    # Take a step
    import numpy as np
    action = np.zeros(env.action_space.shape)
    step_ret = env.step(action)
    print("Step successful! Reward components:", env.reward_info.keys())
    print("Pos reward:", env.reward_info['body_pos_reward'])
    print("Vel reward:", env.reward_info['body_vel_reward'])

if __name__ == "__main__":
    main()
