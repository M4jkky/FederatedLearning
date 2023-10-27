import torch
import torch.optim as optim
import random as random_seed
import numpy as np
import time
from dataset import prepare_dataset
from model import Net, train, test

start_time = time.time()

# Set random seed for reproducibility
seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Hyperparameters
batch_size = 1024
learning_rate = 0.001
epochs = 20
input_size = 7
hidden_size = 16
output_size = 2

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Loading and preprocessing datasets
    train_loader, val_loader, test_loader = prepare_dataset(batch_size)

    # 2. Define model and send to device
    model = Net(input_size, hidden_size, output_size).to(device)

    # 3. Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Train the model on training data
    train(model, train_loader, val_loader, optimizer, epochs, device)

    # 5. Test the model on unseen data
    test(model, test_loader, device)

    # 6. Print time taken to run the program
    end_time = time.time()
    print("Time: ", (end_time - start_time)/60, " seconds")


if __name__ == "__main__":
    main()
