import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from flask import Flask, request, render_template

app = Flask(__name__)


# Load the pre-trained PyTorch model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net(input_size=6, hidden_size=16, output_size=2)
model.load_state_dict(torch.load('/Web/misc/best_model.pth'))
model.eval()

# Load the StandardScaler used during training
scaler = joblib.load('/Web/misc/scaler.pkl')

# Initialize empty lists for storing actual labels and predicted labels
actual_labels = []
predicted_labels = []
correct_predictions = []
incorrect_predictions = []


# Define the routes
@app.route('/')
def index():
    return render_template('templates/index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global correct_predictions, incorrect_predictions, actual_labels, predicted_labels
    # Clear the lists before making new predictions
    actual_labels.clear()
    predicted_labels.clear()
    correct_predictions.clear()
    incorrect_predictions.clear()

    if request.method == 'POST':
        file = request.files['file']
        # Read the uploaded CSV file
        data = pd.read_csv(file)

        data.reset_index(drop=True, inplace=True)
        data['ID'] = data.index + 1

        # Extract relevant features
        features = data[['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']]

        # Scale features using the same StandardScaler instance
        scaled_features = scaler.transform(features)

        # Convert features to PyTorch tensor
        input_tensor = torch.tensor(scaled_features).float()

        # Make predictions for each row
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

            # Process predictions for each record
        for i, prediction in enumerate(predicted):
            actual_label = data['diabetes'][i]
            predicted_label = prediction.item()

            actual_labels.append(actual_label)
            predicted_labels.append(predicted_label)

            if predicted_label == 1 and actual_label == 1:
                correct_predictions.append(data.iloc[i])
            elif predicted_label == 0 and actual_label == 0:
                correct_predictions.append(data.iloc[i])
            else:
                incorrect_predictions.append(data.iloc[i])

    # Return the result messages and predictions to the template
    return render_template('templates/result.html',
                           correct_predictions=correct_predictions,
                           incorrect_predictions=incorrect_predictions,
                           num_correct=len(correct_predictions),
                           num_incorrect=len(incorrect_predictions),
                           accuracy=round(accuracy_score(actual_labels, predicted_labels), 2),
                           precision=round(precision_score(actual_labels, predicted_labels), 2),
                           recall=round(recall_score(actual_labels, predicted_labels), 2),
                           f1=round(f1_score(actual_labels, predicted_labels), 2))


if __name__ == '__main__':
    app.run(debug=True)