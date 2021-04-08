import sys

import numpy as np
from flask import Flask, request, render_template, logging
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('result.html', prediction_text='Employee Monthly Salary should be Rs {}/-'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
