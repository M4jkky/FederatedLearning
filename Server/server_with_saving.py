import torch.nn.functional as F
import random as random_seed
import torch.nn as nn
import numpy as np
import flwr as fl
import torch
import hydra
import os

from typing import List, Tuple, Optional, Dict, Union
from flwr.server.client_proxy import ClientProxy
from omegaconf import DictConfig, OmegaConf
from flwr.common import Parameters, Scalar, FitRes
from collections import OrderedDict

seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

os.makedirs("./models", exist_ok=True)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """
    Calculate the weighted average of the accuracy metric.

    This function takes a list of tuples, where each tuple contains the number of examples and the metrics dictionary.
    It calculates the total number of examples and the weighted sum of the accuracy metric.
    The weighted average is then calculated by dividing the weighted sum by the total number of examples.

    Args:
        metrics (List[Tuple[int, fl.common.Metrics]]): A list of tuples, where each tuple contains the number of examples and the metrics dictionary.

    Returns:
        fl.common.Metrics: A dictionary containing the weighted average of the accuracy metric.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)

    return {"accuracy": weighted_accuracy / total_examples}


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    A custom strategy for the Flower server that extends the FedAvg strategy.
    This strategy not only aggregates the model weights from the clients, but also saves the aggregated model after each round.

    Attributes:
        net: A PyTorch model that will be updated with the aggregated parameters.
    """

    def __init__(self, net, evaluate_metrics_aggregation_fn=None):
        """
        Initialize the SaveModelStrategy.

        Args:
            net: A PyTorch model.
            evaluate_metrics_aggregation_fn: A function to aggregate the evaluation metrics. Defaults to None.
        """
        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.net = net

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights using weighted average and store checkpoint.

        Args:
            server_round: The current round number.
            results: A list of tuples containing the client proxy and the fit result.
            failures: A list of tuples containing the client proxy and the fit result for failed clients, or exceptions.

        Returns:
            A tuple containing the aggregated parameters and the aggregated metrics.
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
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

    # Load the model
    net = Net(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size)

    # Start Flower server with the SaveModelStrategy
    strategy = SaveModelStrategy(net, evaluate_metrics_aggregation_fn=weighted_average)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(cfg.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
