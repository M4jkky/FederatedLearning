import torch.nn.functional as F
import torch.nn as nn
import torch

# Define model / same as centralized
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train the network on train set
def train(net, train_loader, optimizer, num_epochs, device, writer):
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


# Testing the network on test set
def test(net, val_loader, device):
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
