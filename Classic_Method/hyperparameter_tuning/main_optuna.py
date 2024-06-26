import random as random_seed
import torch.optim as optim
import numpy as np
import optuna
import torch
import time

from dataset import prepare_dataset
from model_optuna import Net, train

start_time = time.time()

# Set random seed for reproducibility
seed = 42
random_seed.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the objective function for Optuna
def objective(trial):
    batch_size = trial.suggest_int('batch_size', 8, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 10, 40)
    hidden_size = trial.suggest_int('hidden_size', 4, 256, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'AdamW'])
    momentum = trial.suggest_float('momentum', 0.7, 1.0)

    # 1. Loading and preprocessing Datasets
    train_loader, val_loader, test_loader = prepare_dataset(batch_size)

    # 2. Define model and send to device
    model = Net(input_size=6, hidden_size=hidden_size, output_size=2).to(device)

    # 3. Define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    # 4. Train the model on training data
    f1 = train(model, train_loader, val_loader, optimizer, epochs, device)

    return f1


def main():
    # Create an object and optimize the objective function
    study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="maximize", study_name="more_optimizers")
    study.optimize(objective, n_trials=2)

    # Print the result
    best_trial = study.best_trial
    print(f'Best trial: score {best_trial.value}, params {best_trial.params}')

    # 6. Print time taken to run the program
    print(f"Time: {(time.time() - start_time):.1f} seconds")


if __name__ == "__main__":
    main()
