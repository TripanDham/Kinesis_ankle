import os, sys, torch, numpy as np, hydra
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf, DictConfig
from src.env.myolegs_IL import MyoLegsGAIL
@hydra.main(config_path="../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env = MyoLegsGAIL(cfg)
    obs, _ = env.reset()
    s_len = len(obs)
    s_calc = env.get_obs_size()
    print(f"Calculated: {s_calc}, Actual: {s_len}")
    print(f"Prop inputs: {cfg.run.proprioceptive_inputs}")
    for k, v in env.proprioception.items(): print(f"{k}: {len(v)}")
main()
