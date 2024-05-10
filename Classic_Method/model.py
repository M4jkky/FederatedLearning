import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


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
def train(net, train_loader, val_loader, optimizer, num_epochs, device, writer) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()

    best_val_accuracy = 0.0

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

        # TensorBoard logging for training/validation loss and accuracy
        writer.add_scalar('Loss/train', average_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/val', average_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {average_train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {average_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

        # Update best_val_accuracy if current val_accuracy is higher
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    # Save the model after training is complete
    torch.save(net.state_dict(), f'../Web/misc/model.pth')
    print(f'Saved the best model with validation accuracy: {best_val_accuracy:.2f}')

    print('Training finished.')


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

    cf = confusion_matrix(all_targets, all_predictions)

    # Convert cf and labels to string type
    cf_str = cf.astype(str)
    labels = np.array([['TN', 'FP'], ['FN', 'TP']])

    # Concatenate labels and cf with a newline character in between
    annot = np.core.defchararray.add(np.core.defchararray.add(labels, '\n'), cf_str)

    plt.figure(figsize=(7, 7))
    sns.heatmap(cf, annot=annot, fmt='', cmap='Blues', xticklabels=[], yticklabels=[], cbar=False, square=True)

    plt.xlabel('Skutočné triedy')
    plt.ylabel('Predikované triedy')
    # plt.savefig('Classic_CM.png', dpi=400)
    plt.tight_layout()
    plt.show()

    cr = classification_report(all_targets, all_predictions, target_names=['No Diabetes', 'Diabetes'])
    print(f'Classification Report:\n{cr}')

