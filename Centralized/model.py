import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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
def train(net, train_loader, val_loader, optimizer, num_epochs, device) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for epoch in range(num_epochs):
        best_val_accuracy = 0.0
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
        net.eval()  # Set the model to evaluation mode
        val_total_correct = 0
        val_total_samples = 0
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation during validation
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

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            #torch.save(net.state_dict(), f'model_{best_val_accuracy:.2f}.pth')
            print(f'Saved the model with validation accuracy: {best_val_accuracy:.2f} in epoch {epoch + 1}')

    print('Training finished.')


# Testing the network on test set
def test(net, test_loader, device) -> None:
    correct_predictions = 0
    test_total_samples = 0
    predicted_labels = []
    true_labels = []

    net.eval()
    net.to(device)

    with torch.no_grad():
        for batch in test_loader:
            features, target = batch['features'].to(device), batch['target'].to(device)
            outputs = net(features)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target).sum().item()
            test_total_samples += target.size(0)

            # Store predicted and true labels for calculating metrics
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    test_accuracy = correct_predictions / test_total_samples
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cf = confusion_matrix(true_labels, predicted_labels)

    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    print(f'Confusion Matrix:\n{cf}')
