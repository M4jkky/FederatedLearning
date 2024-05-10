import flwr as fl
import hydra

from omegaconf import DictConfig, OmegaConf
from flwr.common import Metrics
from typing import List, Tuple


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Compute weighted average of the metrics
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)

    return {"accuracy": weighted_accuracy / total_examples}


# Function to start Flower server
def start_flower_server(num_rounds):
    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds),
        strategy=strategy,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Print config for learning
    print(OmegaConf.to_yaml(cfg))

    # 2. Start Flower server for three rounds of federated learning
    start_flower_server(cfg.num_rounds)


if __name__ == "__main__":
    main()
