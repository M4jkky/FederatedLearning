import joblib
import pandas as pd
import torch
from imblearn.over_sampling import KMeansSMOTE
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class DatasetPreprocessing(Dataset):
    def __init__(self, csv_file, transform=None, oversample=True):
        self.data_frame = pd.read_csv(csv_file)

        self.transform = transform

        # Drop 'smoking_history' and 'gender' columns
        self.data_frame = self.data_frame.drop(columns=['smoking_history', 'gender'])

        # Extract features and target columns after dropping 'smoking_history' and 'gender'
        self.features = self.data_frame.drop(columns=['diabetes']).values.astype(float)
        self.target = self.data_frame['diabetes'].values.astype(int)

        # Apply StandardScaler to features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Save the scaler object to a file
        joblib.dump(self.scaler, 'scaler.pkl')

        if oversample:
            kmeans = KMeansSMOTE(cluster_balance_threshold=0.1)
            self.features, self.target = kmeans.fit_resample(self.features, self.target)

        if transform:
            self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'target': self.target[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        features, target = sample['features'], sample['target']
        return {'features': torch.tensor(features, dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.long)}


def prepare_dataset(batch_size, oversample=True):
    try:
        # Load dataset for client 1
        train_dataset = DatasetPreprocessing(csv_file='../Datasets/centralized/train.csv', transform=ToTensor(),
                                             oversample=oversample)
        val_dataset = DatasetPreprocessing(csv_file='../Datasets/centralized/valid.csv', transform=ToTensor())
        test_dataset = DatasetPreprocessing(csv_file='../Datasets/centralized/test.csv', transform=ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)
