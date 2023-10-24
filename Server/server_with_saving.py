from typing import List, Tuple, Optional, Dict, Union
from omegaconf import DictConfig, OmegaConf
from flwr.common import Parameters, Scalar
from collections import OrderedDict
import numpy as np
import flwr as fl
import torch
import hydra
import os

from dataset import prepare_dataset
from model import Net, test


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)

    return {"accuracy": weighted_accuracy / total_examples}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, net: Net):
        super().__init__()
        self.net = net

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from the base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(self.net.state_dict(), f"./models/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


# Define Flower server logic
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print the config
    print(OmegaConf.to_yaml(cfg))

    # Load the test dataset
    test_loader = prepare_dataset(cfg.batch_size)

    # Instantiate your model and move it to the desired device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size).to(device)

    # Start Flower server with the SaveModelStrategy
    strategy = SaveModelStrategy(net)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(cfg.num_rounds),
        strategy=strategy,
    )

    # Evaluate saved models in a for loop based on num_rounds
    for round_number in range(1, cfg.num_rounds + 1):
        # Load the saved model for the current round
        model_path = f"./models/model_round_{round_number}.pth"
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path, map_location=device))
            accuracy = test(net, test_loader, device)
            print(f"Round {round_number} - Test Accuracy: {accuracy:.2f}%")
        else:
            print(f"Model for round {round_number} not found.")


if __name__ == "__main__":
    main()
