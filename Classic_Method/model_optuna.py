from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import torch


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


# Train the network on train set
def train(net, train_loader, val_loader, optimizer, num_epochs, device):
    class_weights = torch.tensor([0.95, 1.1])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    net.to(device)
    net.train()

    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        train_loss = 0.0

        # Training loop
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
            train_loss += loss.item()

        train_accuracy = 100 * total_correct / total_samples
        average_train_loss = train_loss / len(train_loader)

        # Validation loop
        net.eval()
        val_total_correct = 0
        val_total_samples = 0
        val_loss = 0.0

        with torch.no_grad():
            for val_batch in val_loader:
                val_features, val_target = val_batch['features'].to(device), val_batch['target'].to(device)
                val_outputs = net(val_features)
                val_loss += criterion(val_outputs, val_target).item()

                # Calculate accuracy
                _, val_predicted = torch.max(val_outputs, 1)
                val_total_samples += val_target.size(0)
                val_total_correct += (val_predicted == val_target).sum().item()

        val_accuracy = 100 * val_total_correct / val_total_samples
        average_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {average_train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {average_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

    print('Training finished.')

    return val_accuracy


# Testing the network on test set
def test(net, test_loader, device) -> None:
    net.eval()
    net.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features, target = batch['features'].to(device), batch['target'].to(device)
            outputs = net(features)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    cf = confusion_matrix(all_targets, all_predictions)

    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    print(f'Confusion Matrix:\n{cf}')
