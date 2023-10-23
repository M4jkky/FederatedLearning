from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


# Data preprocessing for the CSV file
class DatasetPreprocessing(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = self.data_frame.iloc[idx, 1:].values.astype(float)
        target = int(self.data_frame.iloc[idx, 0])
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
        # Load dataset for client 2
        train_dataset = DatasetPreprocessing(csv_file='/Users/m4jkky/Desktop/school/3rd year/BAKPR/Clients/Datasets/updated_client2', transform=ToTensor())
        val_dataset = DatasetPreprocessing(csv_file='/Users/m4jkky/Desktop/school/3rd year/BAKPR/Clients/Datasets/client2_valid.csv', transform=ToTensor())

        return train_dataset, val_dataset

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)


# Create DataLoader instances
def prepare_dataset(batch_size):
    train_loader, val_dataset = load_dataset()

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
