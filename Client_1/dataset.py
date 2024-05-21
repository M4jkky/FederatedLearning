import pandas as pd
import torch

from imblearn.over_sampling import KMeansSMOTE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class DatasetPreprocessing(Dataset):
    """
    A class used to preprocess the dataset for the machine learning model.

    This class extends PyTorch's Dataset and preprocesses the data by dropping unnecessary columns,
    extracting features and target columns, applying StandardScaler to features, and optionally applying oversampling.

    Attributes:
        data_frame (DataFrame): The DataFrame loaded from the csv file.
        features (ndarray): The features extracted from the DataFrame.
        target (ndarray): The target extracted from the DataFrame.
        scaler (StandardScaler): The StandardScaler used to scale the features.
        transform (callable, optional): An optional transform to be applied on a sample.
    """

    def __init__(self, csv_file, transform=None, oversample=True):
        """
        Initialize the DatasetPreprocessing.

        Args:
            csv_file (str): The path to the csv file.
            transform (callable, optional): An optional transform to be applied on a sample.
            oversample (bool, optional): Whether to apply oversampling.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

        # Drop 'smoking_history' and 'gender' columns and duplicates
        self.data_frame.drop_duplicates(inplace=True)
        self.data_frame = self.data_frame.drop(columns=['smoking_history', 'gender'])

        # Extract features and target columns
        self.features = self.data_frame.drop(columns=['diabetes']).values.astype(float)
        self.target = self.data_frame['diabetes'].values.astype(int)

        # Apply StandardScaler to features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        if oversample:
            kmeans = KMeansSMOTE(cluster_balance_threshold=0.13, sampling_strategy=1.0, kmeans_estimator=45, k_neighbors=8)
            self.features, self.target = (kmeans.fit_resample(self.features, self.target))
        if transform:
            self.transform = transform

        target_df = pd.DataFrame({'diabetes': self.target})
        print(target_df['diabetes'].value_counts())

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the features and target of the sample.
        """
        sample = {'features': self.features[idx], 'target': self.target[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """
    A class used to convert the features and target of a sample to PyTorch tensors.

    This class is callable and can be used as a transform in a Dataset.

    Attributes:
        None
    """

    def __call__(self, sample):
        """
        Convert the features and target of a sample to PyTorch tensors.

        Args:
            sample (dict): A dictionary containing the features and target of a sample.

        Returns:
            dict: A dictionary containing the features and target of the sample as PyTorch tensors.
        """
        features, target = sample['features'], sample['target']
        return {'features': torch.tensor(features, dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.long)}


def prepare_dataset(batch_size, oversample=True):
    """
    Prepare the datasets and data loaders for training and validation.

    This function loads the datasets from csv files, applies the ToTensor transform, and creates DataLoaders.

    Args:
        batch_size (int): The batch size for the DataLoaders.
        oversample (bool, optional): Whether to apply oversampling.

    Returns:
        tuple: A tuple containing the train and validation DataLoaders.

    Raises:
        FileNotFoundError: If the csv file is not found.
        Exception: If any other error occurs.
    """
    try:
        # Load dataset for client 1
        train_dataset = DatasetPreprocessing(csv_file='../Datasets/clients/client1.csv', transform=ToTensor(),
                                             oversample=oversample)
        val_dataset = DatasetPreprocessing(csv_file='../Datasets/clients/valid_c1.csv', transform=ToTensor(),
                                           oversample=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)
