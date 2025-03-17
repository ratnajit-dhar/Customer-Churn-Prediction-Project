from flask import Flask, render_template, request
import pandas as pd
import pickle

with open('ensemble_model.pkl', 'rb') as model:
    model = pickle.load(model)

with open('encoder.pkl', 'rb') as encoder:
    encoders = pickle.load(encoder)

with open('scalers.pkl', 'rb') as scaler:
    scalers = pickle.load(scaler)

app = Flask(__name__)

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        if col=='Churn':
            continue
        input_df[col] = encoder.transform(input_df[col])

    for col, scaler in scalers.items():
        input_df[col] = scaler.transform(input_df[col].values.reshape(-1,1))

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0,1]
    return 'Churn' if prediction == 1 else 'Not Churn', prob

@app.route('/', methods = ['GET', 'POST'])

def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
        }

        prediction, probability = make_prediction(input_data)

    return render_template('index.html', prediction = prediction, probability = probability)

if __name__ == '__main__':

    app.run(debug=True)

