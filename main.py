
import numpy as np
import pickle
app = flask.Flask(__name__)
model1 = pickle.load(open('best_rf_model.pkl', 'rb'))
model2 = pickle.load(open('best_model.pkl', 'rb'))
@app.route('/generalPredict', methods=['POST'])

def model1Api():
    data=flask.request.get_json()
    listOfDics=data["data"]
    answers=[]
    counter=0
    features=10
    for dictionary in listOfDics:
        if "answer" in dictionary:
            answers.append(dictionary["answer"])
            counter+=1
    if counter!=features:
        return flask.jsonify("You didn't answer all the questions! Try again.")
    answers=np.array(answers)
    prediction=model1.Find_disorder(answers) #make the prediction
    return flask.jsonify({"prediction: ",prediction})

@app.route('/depressionPredict', methods=['POST'])
def model2Api():
    data=flask.request.get_json()
    dicsList=data['data']
    answers=[]
    num_of_ans=0
    features=27
    for dictionary in dicsList:
        if 'answer' in dictionary:
           answers.append(dictionary['answer'])
           num_of_ans+=1

    if num_of_ans!=features:
        return flask.jsonify("You didn't answer all the questions! Try again.")
    answers=np.array(answers)
    prediction=model2.find_disorder(answers)
    return flask.jsonify({"prediction: ",prediction})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)
