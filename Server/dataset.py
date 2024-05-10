import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader


# Data preprocessing for the CSV file
class DatasetPreprocessing(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)

        self.transform = transform

        # Drop 'smoking_history' and 'gender' columns and duplicates
        self.data_frame.drop_duplicates(inplace=True)
        self.data_frame = self.data_frame.drop(columns=['smoking_history', 'gender'])

        # Extract features and target columns after dropping 'smoking_history' and 'gender'
        self.features = self.data_frame.drop(columns=['diabetes']).values.astype(float)
        self.target = self.data_frame['diabetes'].values.astype(int)

        # Apply StandardScaler to features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        target_df = pd.DataFrame({'diabetes': self.target})
        print(target_df['diabetes'].value_counts())

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
        # Load test dataset for server
        test_dataset = DatasetPreprocessing(csv_file='dataset/test.csv', transform=ToTensor())

        return test_dataset

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)


# Create DataLoader instances
def prepare_dataset(batch_size):
    test_dataset = load_dataset()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
