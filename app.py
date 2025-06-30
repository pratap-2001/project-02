from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([data])[0]
        return render_template('index.html', prediction_text=f'Predicted class: {prediction}')
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
