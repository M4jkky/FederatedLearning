import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model / same as centralized
class Net(nn.Module):
    """
    A class used to define the neural network model.

    This class extends PyTorch's Module and defines a simple feed-forward neural network with two fully connected layers.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Net.

        Args:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define the forward pass of the network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, train_loader, optimizer, num_epochs, device, writer):
    """
    Train the network on the training set.

    This function trains the network for a specified number of epochs and logs the training loss and accuracy.

    Args:
        net (nn.Module): The neural network model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to train on.
        writer (SummaryWriter): The TensorBoard writer.

    Returns:
        None
    """
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            features, target = batch['features'].to(device), batch['target'].to(device)
            outputs = net(features)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

        accuracy = 100 * total_correct / total_samples
        print(f'Loss: {loss.item():.4f}, Train accuracy: {accuracy:.4f}%')

        # Log training loss and accuracy
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)


def test(net, val_loader, device):
    """
    Test the network on the validation set.

    This function tests the network on the validation set and returns the loss and accuracy.

    Args:
        net (nn.Module): The neural network model to test.
        val_loader (DataLoader): The DataLoader for the validation data.
        device (torch.device): The device to test on.

    Returns:
        tuple: A tuple containing the loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    correct_predictions = 0
    val_total_samples = 0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for batch in val_loader:
            features, target = batch['features'].to(device), batch['target'].to(device)
            outputs = net(features)
            loss = criterion(outputs, target).item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target).sum().item()
            val_total_samples += target.size(0)
            val_accuracy = correct_predictions / val_total_samples

    return loss, val_accuracy
