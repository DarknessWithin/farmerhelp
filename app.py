from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests

# Load your model and label encoder
import joblib

clf = joblib.load('research/crop_model_1.pkl')
le = joblib.load('model/label_encoder.pkl')
crop_feature_means = pd.read_csv('Data/crop_feature_means.csv', index_col=0)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_weather')
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'temperature': '-', 'rainfall': '-'})

    # Use Open-Meteo API
    weather_url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum&timezone=auto'
    res = requests.get(weather_url).json()

    temperature = res.get('current_weather', {}).get('temperature', 'N/A')
    rainfall = res.get('daily', {}).get('precipitation_sum', [0])[0]

    return jsonify({
        'temperature': temperature,
        'rainfall': rainfall
    })


@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        input_data = {
            'Soil_pH': float(request.form['Soil_pH']),
            'Soil_Moisture': float(request.form['Soil_Moisture']),
            'Temperature_C': float(request.form['Temperature_C']),
            'Rainfall_mm': float(request.form['Rainfall_mm']),
            'Fertilizer_Usage_kg': float(request.form['Fertilizer_Usage_kg']),
            'Pesticide_Usage_kg': float(request.form['Pesticide_Usage_kg']),
            'Demand_Index': float(request.form['Demand_Index']),
            'Supply_Index': float(request.form['Supply_Index']),
            'Competitor_Price_per_ton': float(request.form['Competitor_Price_per_ton']),
            'Economic_Indicator': float(request.form['Economic_Indicator']),
            'Weather_Impact_Score': float(request.form['Weather_Impact_Score']),
            'Consumer_Trend_Index': float(request.form['Consumer_Trend_Index']),
        }

        crop, reason = advise_crop_with_reason(input_data)
        return render_template('index.html', prediction_text=f"Recommended Crop: {crop}",
                               reason_text=f"Reason: {reason}")

    except Exception as e:
        return str(e)


def advise_crop_with_reason(agent_input: dict, model=clf, encoder=le):
    features_order = [
        'Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
        'Fertilizer_Usage_kg', 'Pesticide_Usage_kg',
        'Demand_Index', 'Supply_Index',
        'Competitor_Price_per_ton', 'Economic_Indicator',
        'Weather_Impact_Score', 'Consumer_Trend_Index'
    ]

    input_df = pd.DataFrame([agent_input])[features_order]
    prediction_encoded = model.predict(input_df)[0]

    # Fixing inverse_transform
    if hasattr(encoder, 'inverse_transform'):
        prediction_label = encoder.inverse_transform([prediction_encoded])[0]
    else:
        prediction_label = prediction_encoded  # already decoded

    mean_vals = crop_feature_means.loc[prediction_label]
    reasons = []
    for feat in features_order:
        val = agent_input[feat]
        mean_val = mean_vals[feat]
        if (feat in ['Soil_Moisture', 'Rainfall_mm', 'Demand_Index', 'Consumer_Trend_Index']) and val > mean_val:
            reasons.append(f"{feat.replace('_', ' ')} is higher than average for {prediction_label.lower()}")
        elif (feat in ['Temperature_C', 'Supply_Index', 'Pesticide_Usage_kg']) and val < mean_val:
            reasons.append(f"{feat.replace('_', ' ')} is lower, which is ideal for {prediction_label.lower()}")

    explanation = ", ".join(reasons[:3])
    return prediction_label, explanation

@app.route("/news")
def news():
    return render_template("news.html")

import os
NEWS_API_KEY = os.getenv("78c40daf6300471a915c72873876f86c")  # Ensure this environment variable is set

@app.route("/get_news")
def get_news():
    url = ("https://newsapi.org/v2/everything?"
           "q=agriculture&"
           "sortBy=publishedAt&"
           "language=en&"
           f"apiKey={NEWS_API_KEY}")
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])[:5]
    return jsonify([{"title": a["title"], "url": a["url"]} for a in articles])

if __name__ == "__main__":
    app.run(debug=True)
