# Federated Learning
This is a repository for the Federated Learning project for the thesis "Federované učenie v zdravotníctve".  
Working with Diabetes dataset. Which has 100 000 records and 9 attributes. After correlation analysis and
feature selection I decided to use 6 attributes. The dataset is divided into 3 parts. 70% for training, 10% for validation
and 20% for testing -> classic model part. For the federated part I used 2 clients. Server side has 10% of data for testing.
Rest of the data is divided into 2 clients. Each client has 45% of data for training and 5% for validation. Due to imbalance of the dataset I used oversampling technique called KMeansSMOTE.

### Classification report for classic model
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| No Diabetes   | 0.98      | 0.99   | 0.98     | 17991   |
| Diabetes      | 0.84      | 0.74   | 0.79     | 1659    |
| **accuracy**  |           |        | 0.97     | 19650   |
| **macro avg** | 0.91      | 0.86   | 0.88     | 19650   |
| **weighted avg** | 0.96  | 0.97   | 0.97     | 19650   |

### Classification report for federated model (20 epochs, 5 rounds)
|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| No Diabetes   | 0.98      | 0.99   | 0.98     | 18140   |
| Diabetes      | 0.83      | 0.75   | 0.78     | 1660    |
| **accuracy**  |           |        | 0.97     | 19800   |
| **macro avg** | 0.90      | 0.87   | 0.88     | 19800   |
| **weighted avg** | 0.96      | 0.97   | 0.96     | 19800   |


## Technical University of Košice, Intelligent Systems. Department of Cybernetics and Artificial Intelligence 2024.
