import os
import sys
import time
import hydra
import mujoco
from mujoco import viewer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env.myolegs_IL import MyoLegsGAIL

def main():
    with hydra.initialize(config_path="../cfg", version_base="1.1"):
        cfg = hydra.compose(config_name="config", overrides=["env=myolegs_gail", "run=myolegs_gail", "learning=gail_mlp"])
        
    env = MyoLegsGAIL(cfg)
    
    # Reset triggers init_myolegs, which now loads the walk_right keyframe 
    # and overwrites the tracked DOFs with the first frame of the expert motion.
    env.reset()
    
    print("\n" + "="*80)
    print("Environment initialized with HYBRID Pose:")
    print(" - Baseline: 'walk_right' keyframe (for all un-tracked joints)")
    print(" - Overwrite: 10 Joint Angles + 3 Pelvis Angles (from expert Frame 0)")
    print(" - Velocities: ALL ZEROED")
    print("="*80 + "\n")
    
    # Launch viewer passively to just "hold" the pose
    with viewer.launch_passive(env.mj_model, env.mj_data) as v:
        print("Holding position in viewer for 5 seconds...")
        start_time = time.time()
        while time.time() - start_time < 30.0 and v.is_running():
            mujoco.mj_step(env.mj_model, env.mj_data)
            v.sync() # Synchronize the viewer state with mj_data (which is completely static)
            time.sleep(1)
            
    print("Done displaying.")

if __name__ == "__main__":
    main()
