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
        # Load dataset for client 1
        client_1 = DatasetPreprocessing(csv_file='../Datasets/diabetes_client1.csv', transform=ToTensor())
        test_dataset = DatasetPreprocessing(csv_file='../Datasets/diabetes_test.csv', transform=ToTensor())

        return client_1, test_dataset

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)


# Create DataLoader instances
def prepare_dataset(batch_size):
    client_1, test_dataset = load_dataset()
    client_1 = DataLoader(client_1, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_1, test_loader
