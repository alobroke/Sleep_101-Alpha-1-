from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('sleep_quality_model.pkl')

# Mappings
occupation_mapping = {
    'Teacher': 0,
    'Engineer': 1,
    'Doctor': 2,
    'Artist': 3,
    'Scientist': 4,
    'Other': 5
}

bmi_mapping = {
    'Underweight': 0,
    'Normal': 1,
    'Overweight': 2,
    'Obese': 3
}

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    gender = float(request.form['gender'])
    age = float(request.form['age'])
    occupation = request.form['occupation']
    sleep_duration = float(request.form['sleep_duration'])
    physical_activity_level = float(request.form['physical_activity_level'])
    stress_level = float(request.form['stress_level'])
    bmi_category = request.form['bmi_category']

    # Convert to numeric
    occupation_numeric = occupation_mapping.get(occupation, -1)
    bmi_numeric = bmi_mapping.get(bmi_category, -1)

    features = np.array([[gender, age, occupation_numeric, sleep_duration, physical_activity_level, stress_level, bmi_numeric, 0]])  # Placeholder

    prediction = model.predict(features)
    health_score = prediction[0]

    if health_score < 4:
        health_status = "Unhealthy"
    elif 4 <= health_score <= 7:
        health_status = "Average"
    else:
        health_status = "Healthy"

    prediction_text = f'Predicted Quality of Sleep Score: {health_score:.2f}'

    return render_template('index.html', prediction_text=prediction_text, health_status=health_status)

# 🔧 Add this new route for the recommendation page
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

if __name__ == "__main__":
    app.run(debug=True)
