🧩 PROJECT 1: Linear Regression — House Price Prediction
📁 Folder structure
house-price/
├── app.py
├── train.py
├── house.csv
├── requirements.txt
├── Dockerfile
└── .github/workflows/cicd.yml

🧠 train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv('house.csv')  # columns: Area, Rooms, Price
X = data[['Area', 'Rooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R² Score: {r2:.2f}")

joblib.dump((model, r2), 'linear_model.pkl')
print("Model and R² score saved in linear_model.pkl")

🌐 app.py
from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(_name_)
model, r2 = joblib.load('linear_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['Area'], data['Rooms']]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({
        'Predicted Price': round(float(prediction), 2),
        'Model R2 Score': round(r2, 2)
    })

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)

🌳 PROJECT 2: Decision Tree — Iris Flower Classification
🧠 train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('iris.csv')  # columns: sepal_length,sepal_width,petal_length,petal_width,species
X = data.iloc[:, :-1]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

joblib.dump((model, acc), 'tree_model.pkl')
print("Model and accuracy saved in tree_model.pkl")

🌐 app.py
from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(_name_)
model, acc = joblib.load('tree_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({
        'Predicted Species': prediction,
        'Model Accuracy': round(acc, 2)
    })

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)

🌲 PROJECT 3: Random Forest — Diabetes Detection
🧠 train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('diabetes.csv')
X = data[['Glucose', 'Insulin', 'BMI']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

joblib.dump((model, acc), 'forest_model.pkl')
print("Model and accuracy saved in forest_model.pkl")

🌐 app.py
from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(_name_)
model, acc = joblib.load('forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['Glucose'], data['Insulin'], data['BMI']]).reshape(1, -1)
    prediction = model.predict(features)[0]
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return jsonify({
        'Prediction': result,
        'Model Accuracy': round(acc, 2)
    })

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)

🐋 Dockerfile (same for all)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]

⚙ .github/workflows/cicd.yml (simplified for all)
name: CI/CD - ML Model

on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: 3.9
      - run: |
          pip install -r requirements.txt
          python train.py
      - uses: actions/upload-artifact@v4
        with:
          name: model
          path: "*.pkl"

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: model
      - run: |
          docker build -t ml-app .
          docker run -d -p 5000:5000 ml-app

✅ To test locally
python train.py
docker build -t ml-app .
docker run -d -p 5000:5000 ml-app


Then test the endpoint:

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Glucose":120, "Insulin":80, "BMI":25.5}'
