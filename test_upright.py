import numpy as np
from scipy.spatial.transform import Rotation as sRot
import gymnasium as gym
from src.env.myolegs_IL import MyoLegsGAIL
from omegaconf import OmegaConf
cfg = OmegaConf.load("cfg/config.yaml")

env = MyoLegsGAIL(cfg)
env.reset()

print("Initial qpos[3:7]:", env.mj_data.qpos[3:7])
quat = env.mj_data.qpos[3:7]
r = sRot.from_quat([quat[1], quat[2], quat[3], quat[0]])
pelvis_euler = r.as_euler('xyz')
print("pelvis_euler:", pelvis_euler)

# Simulate get_disc_obs logic
upright_trigs = np.array([np.cos(pelvis_euler[0]), np.sin(pelvis_euler[0]), np.cos(pelvis_euler[1]), np.sin(pelvis_euler[1])])
print("upright_trigs:", upright_trigs)
fall_forward = np.angle(upright_trigs[0] + 1j * upright_trigs[1])
fall_sideways = np.angle(upright_trigs[2] + 1j * upright_trigs[3])
print(fall_forward, fall_sideways)
upright_reward = np.exp(-3 * (fall_forward ** 2 + fall_sideways ** 2))
print("upright_reward:", upright_reward)
