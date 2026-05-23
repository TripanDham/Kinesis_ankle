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
    # Check standing height
    q = env.mj_data.qpos.copy()
    mujoco.mj_kinematics(env.mj_model, env.mj_data)
    pelvis_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    xpos = env.mj_data.xpos[pelvis_id]
    print(f"Pelvis xpos: {xpos}")
    print(f"pelvis_ty: {env.mj_data.qpos[1]}")
    print(f"pelvis_tz: {env.mj_data.qpos[2]}")

if __name__ == "__main__":
    main()
