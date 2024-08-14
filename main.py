from flask import Flask, request, jsonify
import numpy as np
import pickle
from Model1 import DisorderPredictor
from Model2 import DisorderPredictor

app = Flask(__name__)
predictor2 = DisorderPredictor('best_rf_model.pkl')
predictor1=DisorderPredictor('best_model.pkl')


@app.route('/generalPredict', methods=['POST'])
def model1Api():
    data = request.get_json()
    listOfDics = data["data"]
    answers = []
    counter = 0
    features = 27
    for dictionary in listOfDics:
        if "answer" in dictionary:
            answers.append(dictionary["answer"])
            counter += 1
    if counter != features:
        return jsonify("You didn't answer all the questions! Try again.")
    answers = np.array(answers)
    prediction = predictor1.find_disorder(answers)  # make the prediction
    return jsonify({"prediction: ", prediction})


@app.route('/depressionPredict', methods=['POST'])
def model2Api():
    data = request.get_json()
    dicsList = data['data']
    answers = []
    num_of_ans = 0
    features = 10
    for dictionary in dicsList:
        if 'answer' in dictionary:
            answers.append(dictionary['answer'])
            num_of_ans += 1

    if num_of_ans != features:
        return jsonify("You didn't answer all the questions! Try again.")
    answers = np.array(answers)
    result = predictor2.predict_disorder(answers)
    return jsonify({"prediction: ", result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
