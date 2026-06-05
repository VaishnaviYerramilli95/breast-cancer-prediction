from flask import Flask, render_template, request
import numpy as np
import joblib

# Create Flask app
app = Flask(__name__)

# Load ML model
model = joblib.load("breast_cancer_model.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods=["POST"])
def predict():
    # Take values from form
    radius = float(request.form["radius"])
    texture = float(request.form["texture"])
    perimeter = float(request.form["perimeter"])
    area = float(request.form["area"])
    smoothness = float(request.form["smoothness"])
    compactness = float(request.form["compactness"])

    # Convert to array
    data = np.array([[radius, texture, perimeter, area, smoothness, compactness]])

    # Predict
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    # Result
    if prediction == 1:
        result = "⚠️ Cancer Detected"
    else:
        result = "✅ Benign (No Cancer)"

    return render_template(
        "index.html",
        result=result,
        probability=round(probability, 2)
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
