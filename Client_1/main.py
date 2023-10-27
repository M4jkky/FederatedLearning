from omegaconf import DictConfig, OmegaConf
import random as random_seed
import numpy as np
import torch
import hydra

from client import generate_client
from dataset import prepare_dataset

seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Print config for learning
    print(OmegaConf.to_yaml(cfg))

    # 2. Loading and preprocessing datasets
    train_loader, val_loader = prepare_dataset(cfg.batch_size)

    # 3. Define clients
    generate_client(train_loader, val_loader, cfg.server_address, cfg)


if __name__ == "__main__":
    main()
