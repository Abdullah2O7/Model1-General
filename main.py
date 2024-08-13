import flask
from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the models
model1 = pickle.load(open('best_rf_model.pkl', 'rb'))
model2 = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/generalPredict', methods=['POST'])
def general_predict():
    data = request.get_json()
    listOfDics = data.get("data", [])
    answers = []
    counter = 0
    features = 10

    for dictionary in listOfDics:
        if "answer" in dictionary:
            answers.append(dictionary["answer"])
            counter += 1

    if counter != features:
        return jsonify({"error": "You didn't answer all the questions! Try again."}), 400

    answers = np.array(answers).reshape(1, -1)  # Reshape for model input
    prediction = model1.predict(answers)
    
    return jsonify({"prediction": prediction.tolist()})

@app.route('/depressionPredict', methods=['POST'])
def depression_predict():
    data = request.get_json()
    dicsList = data.get('data', [])
    answers = []
    num_of_ans = 0
    features = 27

    for dictionary in dicsList:
        if 'answer' in dictionary:
            answers.append(dictionary['answer'])
            num_of_ans += 1

    if num_of_ans != features:
        return jsonify({"error": "You didn't answer all the questions! Try again."}), 400

    answers = np.array(answers).reshape(1, -1)  # Reshape for model input
    prediction = model2.predict(answers)  # Use predict for standard model

    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)
