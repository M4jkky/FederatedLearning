import random as random_seed
import torch.optim as optim
import numpy as np
import torch
import time

from tensorboardX import SummaryWriter
from dataset import prepare_dataset
from model import Net, train, test

start_time = time.time()

# Set random seed for reproducibility
seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

writer = SummaryWriter()

# Hyperparameters
batch_size = 104
learning_rate = 0.02468996615073674
epochs = 31
input_size = 6
hidden_size = 27
output_size = 2

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    # 1. Loading and preprocessing Datasets
    train_loader, val_loader, test_loader = prepare_dataset(batch_size)

    # 2. Define model and send to device
    model = Net(input_size, hidden_size, output_size).to(device)

    # 3. Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Train the model on training data
    train(model, train_loader, val_loader, optimizer, epochs, device, writer)

    # 5. Test the model on unseen data
    test(model, test_loader, device)

    # 6. Print time taken to run the program
    print(f"Time: {(time.time() - start_time):.1f} seconds")

    # 7. Close the SummaryWriter
    writer.close()


if __name__ == "__main__":
    main()
