from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# load model
model = joblib.load('house_price_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get form data
    data = {
        'id': 0, # todo: remove hardcoded id
        'area': int(request.form['area']),
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': int(request.form['bathrooms']),
        'stories': int(request.form['stories']),
        'mainroad': 1 if request.form['mainroad'] == 'yes' else 0,
        'guestroom': 1 if request.form['guestroom'] == 'yes' else 0,
        'basement': 1 if request.form['basement'] == 'yes' else 0,
        'hotwaterheating': 1 if request.form['hotwaterheating'] == 'yes' else 0,
        'airconditioning': 1 if request.form['airconditioning'] == 'yes' else 0,
        'parking': int(request.form['parking']),
        'prefarea': 1 if request.form['prefarea'] == 'yes' else 0,
        'furnished': request.form['furnished']
    }

    # convert to dataframe
    df = pd.DataFrame([data])

    # predict
    prediction = model.predict(df)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)