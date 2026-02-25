import os
import sys
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from src.env.myolegs_IL import MyoLegsIL

def test_env():
    # Dummy config
    cfg = OmegaConf.create({
        "run": {
            "xml_path": "data/xml/myolegs.xml",
            "control_mode": "direct",
            "proprioceptive_inputs": ["root_height", "root_tilt", "local_body_pos", "local_body_rot", "local_body_vel", "local_body_ang_vel"],
            "deactivate_muscles": False,
            "headless": True,
            "fast_forward": True
        },
        "env": {
            "sim_timestep_inv": 150,
            "control_frequency_inv": 5,
            "kp_scale": 1.0,
            "kd_scale": 1.0,
            "reward_specs": {
                "k_pos": 200, "k_vel": 5, "k_energy": 0.05,
                "w_pos": 0.6, "w_vel": 0.2, "w_upright": 0.1, "w_energy": 0.1
            }
        }
    })

    print("Instantiating MyoLegsIL...")
    try:
        env = MyoLegsIL(cfg)
        print("Environment instantiated successfully.")
        
        print(f"Expert data columns: {list(env.ref_motion.keys())}")
        print(f"Expert data length: {len(env.ref_motion['time'])}")
        
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        action = np.zeros(80) # Assuming 80 muscles
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful. Reward: {reward}")
        print(f"Info keys: {list(info.keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()
