from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

# Get model feature columns
model_columns = model.feature_names_in_

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    rainfall = float(request.form["rainfall"])
    pesticides = float(request.form["pesticides"])
    temp = float(request.form["temp"])
    area = request.form["area"]
    item = request.form["item"]

    # Create dataframe
    data = pd.DataFrame({
        "Year":[year],
        "average_rain_fall_mm_per_year":[rainfall],
        "pesticides_tonnes":[pesticides],
        "avg_temp":[temp],
        "Area":[area],
        "Item":[item]
    })

    # One-hot encoding
    data = pd.get_dummies(data)

    # Align columns with model
    data = data.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(data)[0]

    result = round(prediction,2)

    return render_template("index.html",prediction_text=f"Predicted Yield: {result}")

if __name__ == "__main__":
    app.run(debug=True)