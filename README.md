# Federated Learning
This is a repository for the Federated Learning project for the thesis "Federované učenie v zdravotníctve".  
Working with Diabetes dataset. Which has 100 000 records and 9 attributes. After corelation analysis and
feature selection I decided to use 6 attributes. The dataset is divided into 3 parts. 70% for training, 10% for validation
and 20% for testing -> centralized part. For the federated part I used 2 clients. Server side has 10% of data for testing.
Rest of the data is divided into 2 clients. Each client has 45% of data for training and 5% for validation.  
After hyperparameter tuning I achieved about 0.90 accuracy on the test set. Precision 0.95, recall
0.77 and F1 score 0.85. Due to imbalance of the dataset I used oversampling technique called KMeansSMOTE. But I will try
different techniques in the future.  
  
I'm planning to make a web application for this project and also for data visualization.

### Technical University of Košice, Intelligent Systems. Department of Cybernetics and Artificial Intelligence 2023.
