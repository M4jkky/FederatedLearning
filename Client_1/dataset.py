from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
import pandas as pd
import torch


# Data preprocessing for the CSV file
class DatasetPreprocessing(Dataset):
    def __init__(self, csv_file, transform=None):
        # Read the CSV-like data with specified columns
        data = pd.read_csv(csv_file, usecols=['AgeCategory', 'Stroke', 'PhysicalHealth', 'Diabetic',
                                              'DiffWalking', 'KidneyDisease', 'Smoking', 'HeartDisease'])

        print("Data preprocessing from the CSV file...")

        categorical_columns = ['AgeCategory']
        for col in categorical_columns:
            data[col] = data[col].astype('category').cat.codes

        # Convert yes/no columns into binary values
        binary_columns = ['Stroke', 'PhysicalHealth', 'Diabetic', 'DiffWalking',
                          'KidneyDisease', 'Smoking', 'HeartDisease']

        for col in binary_columns:
            data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)

        self.data_frame = data
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = self.data_frame.iloc[idx, 1:].values.astype(float)
        target = self.data_frame.iloc[idx, 0]
        sample = {'features': features, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Convert data to PyTorch tensors
class ToTensor:
    def __call__(self, sample):
        features, target = sample['features'], sample['target']
        return {'features': torch.tensor(features, dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.long)}


# Loading dataset
def load_dataset():
    try:
        # Load dataset for client 1
        train_dataset = DatasetPreprocessing(csv_file='/Users/m4jkky/Desktop/school/3rd year/BAKPR/Clients/Datasets/heart/client1_valid.csv', transform=ToTensor())
        val_dataset = DatasetPreprocessing(csv_file='/Users/m4jkky/Desktop/school/3rd year/BAKPR/Clients/Datasets/heart/heart_client1.csv', transform=ToTensor())
        return train_dataset, val_dataset

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)


# Create DataLoader instances
def prepare_dataset(batch_size):
    train_dataset, val_dataset = load_dataset()

    # Extract features and targets from the training dataset
    X_train = [sample['features'] for sample in train_dataset]
    y_train = [sample['target'] for sample in train_dataset]

    # Apply SMOTE oversampling to balance the class distribution
    print("Applying SMOTE oversampling to balance the class distribution...")
    print("\n")
    smote = SMOTE(sampling_strategy=0.7)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Wrap the resampled data into a custom dataset
    class ResampledDataset(Dataset):
        def __init__(self, features, targets):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.long)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return {'features': self.features[idx], 'target': self.targets[idx]}

    # Create DataLoader instances for resampled data
    resampled_train_dataset = ResampledDataset(X_train_resampled, y_train_resampled)
    train_loader = DataLoader(resampled_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader