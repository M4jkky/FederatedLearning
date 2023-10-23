import random as random_seed
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from client import generate_client
from dataset import prepare_dataset


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Print config for learning
    print(OmegaConf.to_yaml(cfg))

    # 2. Loading and preprocessing datasets
    trainloaders, testloader = prepare_dataset(cfg.batch_size)

    # 3. Define clients
    server_address = "127.0.0.1:8080"
    generate_client(trainloaders, testloader, server_address, cfg)


if __name__ == "__main__":
    main()
