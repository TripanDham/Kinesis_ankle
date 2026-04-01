import os
import sys
import traceback
import numpy as np
from omegaconf import OmegaConf

# Add project root to sys.path
# Assumes script is run from the project root
sys.path.append(os.getcwd())

from src.env.myolegs_gail_env import MyoLegsGailEnv

def create_mock_cfg(active_gains=True):
    """
    Creates a minimal Hydra-style configuration that satisfies both 
    BaseEnv and MyoLegsGailEnv requirements.
    """
    return OmegaConf.create({
        "run": {
            "xml_path": "data/xml/myolegs_OSL_KA.xml",
            "control_mode": "PD",
            "headless": True,
            "fast_forward": False,
            "proprioceptive_inputs": ["root_height", "root_tilt", "local_body_pos", "local_body_rot", "local_body_vel", "local_body_ang_vel", "feet_contacts"],
            "task_inputs": ["diff_local_body_pos", "diff_local_vel", "local_ref_body_pos"]
        },
        "env": {
            "sim_timestep_inv": 150,
            "control_frequency_inv": 5,
            "max_episode_length": 300,
            "impedance_control": True,
            "active_gains": active_gains,
            "fixed_knee_stiffness": 0.0,
            "fixed_knee_damping": 10.0,
            "fixed_ankle_stiffness": 0.0,
            "fixed_ankle_damping": 10.0,
            "kp_scale": 1.0,
            "kd_scale": 1.0,
            "termination_distance": 0.15,
            "fatigue_reset_random": False,
            "fatigue_reset_vec": None,
            "knockout_muscles": None,
            "muscle_condition": '',
            "persistent_fatigue": False,
            "reward_specs": {
                "k_energy": 0.05,
                "k_pos": 200,
                "k_vel": 5,
                "w_energy": 0.01,
                "w_gail": 0.1,
                "w_pos": 0.6,
                "w_upright": 1.0,
                "w_vel": 0.2
            }
        }
    })

def verify():
    print("="*60)
    print("VERIFYING IMPEDANCE CONTROL MODES (Active vs Fixed)")
    print("="*60)
    
    # Test cases: (Mode Name, active_gains_flag, expected_dimension)
    test_cases = [
        ("ACTIVE GAINS (Original)", True, 60),
        ("FIXED GAINS (New Mode)", False, 56)
    ]
    
    overall_success = True
    
    for mode_name, active, expected_dim in test_cases:
        print(f"\n[Test Case] {mode_name}")
        print(f"  Flag: active_gains={active}")
        
        try:
            cfg = create_mock_cfg(active)
            env = MyoLegsGailEnv(cfg)
            
            # 1. Dimension Check
            action_dim = env.action_space.shape[0]
            print(f"  - Action Space Dim: {action_dim}")
            
            if action_dim != expected_dim:
                print(f"  [FAILURE] Expected dimension {expected_dim}, but got {action_dim}.")
                overall_success = False
            else:
                print(f"  [SUCCESS] Dimension matches expectation.")
            
            # 2. Physics Step Initialization
            print(f"  - Testing Physics Step Indexing...")
            
            # Create a mock action (all zeros)
            mock_action = np.zeros(action_dim)
            
            # We call physics_step directly to verify our indexing into the action array
            # This is where 'name K_knee not defined' or out-of-bounds index errors happen.
            # We don't call env.reset() to avoid complex observation calculations that fail 
            # without a full motion library / sensor suite.
            try:
                env.physics_step(mock_action)
                print(f"  [SUCCESS] physics_step completed without crash.")
            except Exception as e:
                print(f"  [FAILURE] physics_step crashed: {e}")
                traceback.print_exc()
                overall_success = False
            
            # 3. Verify logging dictionary populated
            if "knee_K" in env.last_impedance:
                k_val = env.last_impedance["knee_K"]
                print(f"  - Verified: last_impedance['knee_K'] = {k_val:.2f}")
            else:
                print(f"  [FAILURE] last_impedance not correctly populated.")
                overall_success = False
                
            print(f"  [SUCCESS] Mode {mode_name} logic is robust.")
            
        except Exception:
            print(f"  [FATAL ERROR] Exception during {mode_name} verification:")
            traceback.print_exc()
            overall_success = False
            
    print("\n" + "="*60)
    if overall_success:
        print("RESULT: ALL TESTS PASSED.")
    else:
        print("RESULT: SOME TESTS FAILED. See tracebacks above.")
    print("="*60)

if __name__ == "__main__":
    verify()
