

import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('ineuron_bank', 'rb'))
scaler = pickle.load(open('scaling','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()  # Use get_json() to extract JSON data from the request
    print(data)

    # Check if all required features are present in the input data
    required_features = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                         'savings', 'personal_status_sex','property', 'age', 'people_liable','installment_rate', 
                         'other_installment_plans','housing', 'other_debtors','employment_duration','number_credits', 
                         'job', 'present_residence','telephone', 'foreign_worker']

    if not all(feature in data['data'] for feature in required_features):
        return jsonify({"error": "Missing required features"}), 400

    # Extract the features from the input data
    new_data = scaler.transform(np.array([data['data'][feature] for feature in required_features]).reshape(1, -1))

    # Predict the credit risk using the model
    output = model.predict(new_data)
    print("Prediction:", output)

    output = int(output[0])  # Convert int32 to int

    return jsonify({"credit_risk": output})


@app.route('/predict', methods=['POST'])
def predict():
    data = [int(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)
    credit_risk = int(output[0])

    # Determine the message based on the credit risk prediction
    if credit_risk == 1:
        risk_message = "a Good Risk."
    else:
        risk_message = "a Bad Risk."

    return render_template('home.html', prediction_text="It is {}".format(risk_message))


if __name__ == '__main__':
    app.run(debug=True)
