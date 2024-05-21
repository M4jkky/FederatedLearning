import flwr as fl
import torch
import yaml

from collections import OrderedDict
from model import Net, train, test


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, config, writer) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_model_params = None
        self.best_train_accuracy = 0.0
        self.writer = writer

        self.model = Net(input_size=config['input_size'], hidden_size=config['hidden_size'],
                         output_size=config['output_size'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.
        that belongs to this client. Then, send it back to the server."""
        with open('./conf/config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        train(self.model, self.train_loader, optimizer, num_epochs=config['local_epochs'], device=self.device,
              writer=self.writer)

        return self.get_parameters({}), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data held by this client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.val_loader, device=self.device)

        return float(loss), len(self.val_loader), {"accuracy": accuracy}


def generate_client(train_loaders, test_loaders, server_address, config, writer):
    """Generate a Flower client for each dataset partition."""
    fl.client.start_numpy_client(
        server_address=server_address,
        client=Client(train_loaders, test_loaders, config, writer)
    )
