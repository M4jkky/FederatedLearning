import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Test the network on server test/valid dataset
def test(net, test_loader, device):
    correct_predictions = 0
    test_total_samples = 0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for batch in test_loader:
            features, target = batch['features'].to(device), batch['target'].to(device)
            outputs = net(features)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target).sum().item()
            test_total_samples += target.size(0)
            test_accuracy = correct_predictions / test_total_samples
    return test_accuracy
