from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = request.form['sex']
    fare = float(request.form['fare'])

    sex = 0 if sex.lower() == 'male' else 1

    data = np.array([[age, sex, fare]])
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = " 🎉 Congratulatons !! Survived"
        result_class = "survived"
    else:
        result = " ⚠️ Alert !! Not Survived"
        result_class = "not-survived"

    return render_template('predict.html', prediction_text=f"Prediction: {result}", result_class=result_class)

if __name__ == '__main__':
    app.run(debug=True)
