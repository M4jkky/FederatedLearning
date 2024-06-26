import random as random_seed
import numpy as np
import torch
import hydra

from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from dataset import prepare_dataset
from client import generate_client

seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

writer = SummaryWriter()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Print config for learning
    print(OmegaConf.to_yaml(cfg))

    # 2. Loading and preprocessing Datasets
    train_loader, val_loader = prepare_dataset(cfg.batch_size)

    # 3. Define clients
    generate_client(train_loader, val_loader, cfg.server_address, cfg, writer)

    # 4. Close the SummaryWriter
    writer.close()


if __name__ == "__main__":
    main()
