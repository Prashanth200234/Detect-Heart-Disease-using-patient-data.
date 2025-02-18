import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from flask import Flask, render_template, request

# Load dataset
df = pd.read_csv('dataset.csv')  # Update with actual file name

# Data Preprocessing
df = df.dropna()

# Convert categorical features if necessary (e.g., encoding chest pain type)
df = pd.get_dummies(df, columns=['chest pain type', 'resting ecg', 'ST slope'], drop_first=True)

X = df.drop(columns=['target'])  # Features
y = df['target']  # Target variable

# Normalize numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Testing
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save Model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

import os

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    data = [float(x) for x in request.form.values()]
    
    # Create a DataFrame with the same columns as the training set
    df_input = pd.DataFrame([data], columns=['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                                             'fasting blood sugar', 'resting ecg', 'max heart rate', 
                                             'exercise angina', 'oldpeak', 'ST slope'])

    # Apply one-hot encoding to match training format
    df_input = pd.get_dummies(df_input, columns=['chest pain type', 'resting ecg', 'ST slope'])

    # Ensure missing columns from training are added with default value 0
    missing_cols = [col for col in X.columns if col not in df_input.columns]
    for col in missing_cols:
        df_input[col] = 0

    # Reorder columns to match training data
    df_input = df_input[X.columns]

    # Scale input data
    final_data = scaler.transform(df_input)

    # Predict
    prediction = model.predict(final_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return render_template('index.html', prediction_text=result)



if __name__ == '__main__':
    app.run(debug=True)
