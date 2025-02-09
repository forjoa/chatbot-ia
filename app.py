from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('house_price_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatpredict', methods=['POST'])
def chatpredict():
    data = request.get_json()
    
    # Convertir los datos recibidos al formato esperado por el modelo
    data_dict = {
        'id': 0,  # se puede mejorar quitando este campo si no es necesario
        'area': int(data.get('area')),
        'bedrooms': int(data.get('bedrooms')),
        'bathrooms': int(data.get('bathrooms')),
        'stories': int(data.get('stories')),
        'mainroad': 1 if data.get('mainroad', '').strip().lower() == 'yes' else 0,
        'guestroom': 1 if data.get('guestroom', '').strip().lower() == 'yes' else 0,
        'basement': 1 if data.get('basement', '').strip().lower() == 'yes' else 0,
        'hotwaterheating': 1 if data.get('hotwaterheating', '').strip().lower() == 'yes' else 0,
        'airconditioning': 1 if data.get('airconditioning', '').strip().lower() == 'yes' else 0,
        'parking': int(data.get('parking')),
        'prefarea': 1 if data.get('prefarea', '').strip().lower() == 'yes' else 0,
        'furnished': data.get('furnished')
    }
    
    # Convertir a DataFrame
    df = pd.DataFrame([data_dict])
    
    # Realizar la predicción
    prediction = model.predict(df)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
