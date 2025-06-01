from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load data and model
car = pd.read_csv('Cleaned_car.csv')
with open('LinearRegressionModel.pkl', 'rb') as f:   # <-- updated filename
    model = pickle.load(f)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    # Create mapping from company to models
    company_model_map = car.groupby('company')['name'].apply(list).to_dict()
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
        company_model_map=company_model_map
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    company = data['company']
    car_model = data['car_model']
    year = int(data['year'])
    fuel_type = data['fuel_type']
    km_driven = int(data['km_driven'])

    # Prepare input for model (ensure order matches training)
    input_df = pd.DataFrame([[car_model, company, year, km_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Predict using the loaded model
    try:
        predicted_price = model.predict(input_df)[0]
        predicted_price = int(predicted_price)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)

import pickle
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)