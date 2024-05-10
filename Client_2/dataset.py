import pandas as pd
import torch
from imblearn.over_sampling import KMeansSMOTE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class DatasetPreprocessing(Dataset):
    def __init__(self, csv_file, transform=None, oversample=True):
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

        if oversample:
            kmeans = KMeansSMOTE(cluster_balance_threshold=0.13, sampling_strategy=1.0, kmeans_estimator=45,
                                 k_neighbors=8)
            self.features, self.target = (kmeans.fit_resample(self.features, self.target))
        if transform:
            self.transform = transform

        target_df = pd.DataFrame({'diabetes': self.target})
        print(target_df['diabetes'].value_counts())

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
        # Load dataset for client 2
        train_dataset = DatasetPreprocessing(csv_file='../Datasets/clients/client2.csv', transform=ToTensor(),
                                             oversample=oversample)
        val_dataset = DatasetPreprocessing(csv_file='../Datasets/clients/valid_c2.csv', transform=ToTensor(),
                                           oversample=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)
