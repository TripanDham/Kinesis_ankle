import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from src.env.myolegs_IL import MyoLegsGAIL
import mujoco

@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    cfg.run.xml_path = "/media/tripan/Data/DDP/Kinesis_ankle/data/xml/myoLeg26_OSL_A.xml"
    cfg.no_log = True
    
    print("Instantiating MyoLegsGAIL...")
    env = MyoLegsGAIL(cfg)
    
    stand_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
    print(f"\nSearching for 'stand' keyframe id: {stand_id}")
    print(f"Model nq: {env.mj_model.nq}, Keyframe qpos length: {len(env.mj_model.key_qpos[0]) if env.mj_model.nkey > 0 else 0}")
    
    for i in range(env.mj_model.njnt):
        print(f"Jnt {i}: {env.mj_model.joint(i).name} (adr: {env.mj_model.joint(i).qposadr[0]})")
    
    print("\n--- Initial State ---")
    obs, info = env.reset()
    qpos = env.mj_data.qpos.copy()
    print(f"qpos[0:7]: {qpos[0:7]}")
    
    height_val = qpos[2]
    print(f"Current qpos[2] (presumed height): {height_val}")
    
    fell, truncated = env.compute_reset()
    print(f"Initial compute_reset: fell={fell}, truncated={truncated}")
    
    if fell:
        print("\n[ALERT] Environment is resetting immediately because fell=True")
        if qpos[2] < 0.5:
             print("[TIP] Height is at index 2 (pelvis_tz). Value is too low.")

if __name__ == "__main__":
    main()
