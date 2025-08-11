import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- Configuration ---
# NOTE: Download the dataset from https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
# and update this path.
DATA_PATH = "../input/heart-disease-data/heart_disease_uci.csv"
NN_EPOCHS = 100
NN_LR = 0.001

# --- 1. Data Preparation ---
def load_and_prepare_data():
    """Loads and preprocesses the Heart Disease UCI dataset."""
    data = pd.read_csv(DATA_PATH)
    
    # Basic cleaning and one-hot encoding
    data = data.drop(['id', 'dataset'], axis=1)
    data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # Fill missing values with the mean of the column
    for col in data.columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].mean(), inplace=True)

    X = data.drop('num', axis=1)
    y = (data['num'] > 0).astype(int) # Binary classification: 0 = no disease, 1 = disease

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns

# --- 2. Neural Network Component ---
class HeartDiseaseNN(nn.Module):
    """A neural network to predict the probability of heart disease."""
    def __init__(self, input_size):
        super(HeartDiseaseNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

def train_nn(X_train, y_train):
    """Trains the neural network and returns the trained model."""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    
    model = HeartDiseaseNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)
    
    print("\n--- Training Neural Network ---")
    for epoch in range(NN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{NN_EPOCHS}], Loss: {loss.item():.4f}')
    print("--- NN Training Finished ---\n")
    return model

# --- 3. Fuzzy Logic Component ---
def create_fuzzy_system():
    """Creates and configures the fuzzy logic control system."""
    # Define fuzzy variables (Antecedents)
    age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
    chol = ctrl.Antecedent(np.arange(100, 401, 1), 'cholesterol')
    nn_prob = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'nn_probability')

    # Define output variable (Consequent)
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # Define membership functions
    age.automf(3, names=['young', 'middle', 'senior'])
    chol.automf(3, names=['low', 'normal', 'high'])
    nn_prob.automf(3, names=['low', 'medium', 'high'])
    
    risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
    risk['medium'] = fuzz.trimf(risk.universe, [25, 50, 75])
    risk['high'] = fuzz.trimf(risk.universe, [50, 75, 100])

    # Define fuzzy rules
    rule1 = ctrl.Rule(age['senior'] & chol['high'], risk['high'])
    rule2 = ctrl.Rule(nn_prob['high'] | age['senior'], risk['high'])
    rule3 = ctrl.Rule(nn_prob['medium'] & chol['normal'], risk['medium'])
    rule4 = ctrl.Rule(nn_prob['low'] & age['young'], risk['low'])
    rule5 = ctrl.Rule(chol['low'], risk['low'])

    # Create control system
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)
    
    return risk_simulation

# --- 4. Hybrid System Evaluation ---
def evaluate_systems(nn_model, fuzzy_system, X_test, y_test, feature_names):
    """Evaluates the standalone NN and the hybrid system."""
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Standalone NN evaluation
    nn_model.eval()
    with torch.no_grad():
        nn_outputs = nn_model(X_test_tensor).squeeze()
        nn_predictions = (nn_outputs > 0.5).int().numpy()
    nn_accuracy = accuracy_score(y_test, nn_predictions)
    print(f"Standalone Neural Network Accuracy: {nn_accuracy*100:.2f}%")

    # Hybrid system evaluation
    hybrid_predictions = []
    # Get indices for features used by the fuzzy system
    age_idx = list(feature_names).index('age')
    chol_idx = list(feature_names).index('chol')

    for i in range(X_test.shape[0]):
        # Denormalize values for fuzzy system (approximate)
        # This is a simplification; a more robust solution would save the scaler's mean/std
        age_val = X_test[i, age_idx] * 10 + 55 # Approximate denormalization
        chol_val = X_test[i, chol_idx] * 50 + 250 # Approximate denormalization
        
        # Clamp values to be within the fuzzy universe range
        age_val = np.clip(age_val, 20, 80)
        chol_val = np.clip(chol_val, 100, 400)
        
        # Set inputs for the fuzzy system
        fuzzy_system.input['age'] = age_val
        fuzzy_system.input['cholesterol'] = chol_val
        fuzzy_system.input['nn_probability'] = nn_outputs[i].item()
        
        # Compute the fuzzy result
        fuzzy_system.compute()
        risk_score = fuzzy_system.output['risk']
        
        # Classify based on risk score (threshold of 50)
        hybrid_predictions.append(1 if risk_score > 50 else 0)
        
    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
    print(f"Hybrid Neuro-Fuzzy System Accuracy: {hybrid_accuracy*100:.2f}%")

# --- 5. Main Execution ---
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Train the Neural Network
    trained_nn = train_nn(X_train, y_train)
    
    # Create the Fuzzy System
    fuzzy_sim = create_fuzzy_system()
    
    # Evaluate both systems
    evaluate_systems(trained_nn, fuzzy_sim, X_test, y_test, feature_names)

    # Example of a single prediction
    print("\n--- Example Single Prediction ---")
    sample_idx = 0
    sample_X = X_test[sample_idx]
    sample_y = y_test[sample_idx]

    # Get NN prediction
    nn_prob_output = trained_nn(torch.FloatTensor(sample_X)).item()
    
    # Get Fuzzy prediction
    age_val = np.clip(sample_X[list(feature_names).index('age')] * 10 + 55, 20, 80)
    chol_val = np.clip(sample_X[list(feature_names).index('chol')] * 50 + 250, 100, 400)
    
    fuzzy_sim.input['age'] = age_val
    fuzzy_sim.input['cholesterol'] = chol_val
    fuzzy_sim.input['nn_probability'] = nn_prob_output
    fuzzy_sim.compute()
    final_risk = fuzzy_sim.output['risk']

    print(f"Patient Data (scaled): {sample_X[:4]}...")
    print(f"Actual Diagnosis: {'Disease' if sample_y == 1 else 'No Disease'}")
    print(f"NN Probability Output: {nn_prob_output:.3f}")
    print(f"Final Hybrid Risk Score: {final_risk:.2f}%")
    print(f"Hybrid System Prediction: {'Disease' if final_risk > 50 else 'No Disease'}")
