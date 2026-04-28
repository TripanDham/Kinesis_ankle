import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from src.agents.agent_gail import AgentGAIL

logger = logging.getLogger(__name__)

@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    # Disable file logging if in test mode to avoid cluttering/overwriting server logs
    if cfg.run.test:
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logging.root.removeHandler(handler)
        logger.info("Running in TEST mode: File logging disabled.")

    # Setup Device
    device = torch.device(cfg.run.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    
    logger.info(f"Using device: {device}")
    
    if not cfg.no_log:
        import wandb
        wandb.init(
            project=cfg.project,
            name=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            notes=cfg.get("notes", ""),
            mode="online"
        )
    
    # Initialize Agent
    agent = AgentGAIL(
        cfg=cfg,
        dtype=dtype,
        device=device,
        training=not cfg.run.test,
        checkpoint_epoch=cfg.run.get("checkpoint", 0)
    )

    # Inject normalizer into environment for state-wide OOB penalty
    if hasattr(agent.env, 'set_normalizer'):
        agent.env.set_normalizer(agent.policy_net.norm)
    
    # Run
    if cfg.run.test:
        logger.info("Starting Evaluation Pipeline")
        agent.eval_policy(runs=cfg.run.get("num_eval_runs", 10))
    else:
        logger.info("Starting Training Pipeline")
        agent.optimize_policy()
    
    # Cleanup
    if hasattr(agent, "env"):
        agent.env.close()

if __name__ == "__main__":
    main()
