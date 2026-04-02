import os
import sys
import numpy as np
from omegaconf import OmegaConf

# Add current directory to path
sys.path.append(os.getcwd())

from src.env.myolegs_IL import MyoLegsGAIL

def create_mock_motion(path):
    import joblib
    import numpy as np
    # Create a dummy motion with 100 frames
    # The size doesn't strictly matter for bypass, but should be reasonable
    qpos_size = 60 
    qvel_size = 60
    
    # Needs to be a dict of dicts where each inner dict has qpos, qvel
    mock_data = {
        "mock_motion": {
            "qpos": np.zeros((100, qpos_size), dtype=np.float32),
            "qvel": np.zeros((100, qvel_size), dtype=np.float32),
            "fps": 30,
            "trans_orig": np.zeros((100, 3), dtype=np.float32)
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(mock_data, path)
    print(f"Created mock motion file at {path}")

def verify():
    print("--- Verifying Ankle-Only Control Implementation ---")
    
    mock_motion_path = "data/mock_motion.pkl"
    create_mock_motion(mock_motion_path)
    
    # Minimal config to load the env
    cfg = OmegaConf.create({
        "run": {
            "xml_path": "data/xml/myoLeg26_OSL_A.xml",
            "control_mode": "PD",
            "proprioceptive_inputs": ["root_height", "root_tilt", "local_body_pos", "local_body_rot", "local_body_vel", "local_body_ang_vel", "feet_contacts"],
            "task_inputs": ["diff_local_body_pos", "diff_local_vel", "local_ref_body_pos"],
            "deactivate_muscles": False,
            "headless": True,
            "fast_forward": True,
            "num_motions": 1,
            "multi_thread": False,
            "randomize_heading": False,
            "random_sample": False,
            "random_start": False,
            "motion_id": 0,
            "im_eval": False,
            "test": True,
            "motion_file": mock_motion_path,
            "initial_pose_file": "data/initial_pose/initial_pose_train.pkl",
            "device": "cpu",
            "history_len": 1,
            "recording_biomechanics": False
        },
        "env": {
            "max_episode_length": 100,
            "sim_timestep_inv": 150,
            "control_frequency_inv": 5,
            "kp_scale": 1.0,
            "kd_scale": 1.0,
            "impedance_control": True,
            "active_gains": True,
            "termination_distance": 0.15,
            "reward_specs": {
                "k_pos": 200, "k_vel": 5, "k_energy": 0.05,
                "w_pos": 0.6, "w_vel": 0.2, "w_upright": 0.1, "w_energy": 0.1,
                "w_ankle_delta": 0.5 # High weight for visibility
            }
        }
    })

    print("Instantiating environment...")
    try:
        env = MyoLegsGAIL(cfg)
        print(f"Environment Loaded.")
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space Shape: {env.observation_space.shape}")
        
        # Action size should be 25 (22 muscles + 3 ankle params)
        expected_action_size = len(env.muscle_idx) + 3
        print(f"Number of muscles: {len(env.muscle_idx)}")
        print(f"Expected action size: {expected_action_size}")
        
        if env.action_space.shape[0] != expected_action_size:
            print(f"FAILURE: Action space size {env.action_space.shape[0]} != {expected_action_size}")
            return

        env.reset()
        
        print("\nRunning 25 steps to verify latching...")
        last_latched = None
        for i in range(25):
            # Constant muscle actions, changing ankle parameters
            # Let's make ankle params change every step to see if latching works
            action = np.zeros(expected_action_size)
            action[-3:] = [0.1 * i, 0.05 * i, -0.02 * i] # K, B, th changing every step
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            curr_latched = env.latched_ankle_action
            delta = env.delta_ankle_action
            penalty = info.get("ankle_delta_penalty", 0)
            
            if i % 10 == 0:
                print(f"Step {i:2d} (UPDATE): Latched={curr_latched}, Delta={delta}, Penalty={penalty:.4f}")
                if i > 0 and penalty == 0:
                    print("FAILURE: Penalty should be non-zero on update step.")
            else:
                # On non-update steps, latched should remain same as previous step
                # and delta should be zero.
                if not np.array_equal(curr_latched, last_latched):
                    print(f"FAILURE: Step {i:2d} latched action changed! {curr_latched} != {last_latched}")
                if not np.array_equal(delta, np.zeros(3)):
                    print(f"FAILURE: Step {i:2d} delta_ankle_action is non-zero! {delta}")
                if penalty != 0:
                    print(f"FAILURE: Step {i:2d} penalty is non-zero! {penalty}")
            
            last_latched = curr_latched.copy()

        print("\nVerification Complete.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
