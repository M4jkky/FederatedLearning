import random as random_seed
import torch.optim as optim
import numpy as np
import optuna
import torch
import time

from dataset import prepare_dataset
from model_optuna import Net, train, test

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
    # Suggest values for the hyperparameters
    batch_size = trial.suggest_int('batch_size', 8, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 10, 300)
    hidden_size = trial.suggest_int('hidden_size', 4, 256, log=True)

    # 1. Loading and preprocessing Datasets
    train_loader, val_loader, test_loader = prepare_dataset(batch_size)

    # 2. Define model and send to device
    model = Net(input_size=6, hidden_size=hidden_size, output_size=2).to(device)

    # 3. Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Train the model on training data
    val_accuracy = train(model, train_loader, val_loader, optimizer, epochs, device)

    return val_accuracy


def main():
    # Create a study object and optimize the objective function
    study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="maximize", study_name="adam_optimizer")
    study.optimize(objective, n_trials=100)

    # Print the result
    best_trial = study.best_trial
    print(f'Best trial: score {best_trial.value}, params {best_trial.params}')

    # 5. Test the model on unseen data with best trial parameters
    batch_size = best_trial.params['batch_size']
    hidden_size = best_trial.params['hidden_size']
    _, _, test_loader = prepare_dataset(batch_size)
    model = Net(input_size=6, hidden_size=hidden_size, output_size=2).to(device)
    test(model, test_loader, device)

    # 6. Print time taken to run the program
    print(f"Time: {(time.time() - start_time):.1f} seconds")


if __name__ == "__main__":
    main()
