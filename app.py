from flask import Flask, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('best_rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Hello from first model"


if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5001)