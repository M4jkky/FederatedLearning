import pandas as pd
import joblib
import torch

from imblearn.over_sampling import KMeansSMOTE
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class DatasetPreprocessing(Dataset):
    def __init__(self, csv_file, transform=None, oversample=None):
        self.df = pd.read_csv(csv_file)

        # Drop 'smoking_history' and 'gender' columns and duplicates
        self.df.drop_duplicates(inplace=True)
        self.df = self.df.drop(columns=['smoking_history', 'gender'])

        # Extract features and target columns after dropping 'smoking_history' and 'gender'
        self.features = self.df.drop(columns=['diabetes']).values.astype(float)
        self.target = self.df['diabetes'].values.astype(int)

        # Apply StandardScaler to features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Save the scaler object to a file
        joblib.dump(self.scaler, '../Web/misc/scaler.pkl')

        if oversample:
            kmeans = KMeansSMOTE(cluster_balance_threshold=0.13, sampling_strategy=1.0, kmeans_estimator=45, k_neighbors=8)
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


def prepare_dataset(batch_size):
    try:
        # Load dataset
        train_dataset = DatasetPreprocessing(csv_file='../Datasets/classic_method/train.csv', transform=ToTensor(),
                                             oversample=True)
        val_dataset = DatasetPreprocessing(csv_file='../Datasets/classic_method/valid.csv', transform=ToTensor(),
                                           oversample=False)
        test_dataset = DatasetPreprocessing(csv_file='../Datasets/classic_method/test.csv', transform=ToTensor(),
                                            oversample=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)
