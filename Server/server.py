from flwr.common import Metrics
from typing import List, Tuple
import flwr as fl

from omegaconf import DictConfig, OmegaConf
import hydra

from model import Net, test
from dataset import prepare_dataset


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Compute weighted average of the metrics
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)

    return {"accuracy": weighted_accuracy / total_examples}


# def load_and_evaluate_best_model(test_loader):


#    # Evaluate the best model on the server test dataset
#    accuracy = test(best_model, test_loader, device="cpu")
#    print(f"Test Accuracy: {accuracy * 100:.2f}%")


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

    # 2. Prepare the dataset
    test_loader = prepare_dataset(cfg.batch_size)

    # 3. Start Flower server for three rounds of federated learning
    start_flower_server(cfg.num_rounds)

    # 4. Evaluate the best model on the server test dataset
    # load_and_evaluate_best_model(test_loader)


if __name__ == "__main__":
    main()
