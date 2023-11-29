import io
import flask
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import seaborn as sns

app = Flask(__name__)

model = load_model('model.h5')
model_2 = LogisticRegression()
dataset = np.load('dataset.npz')
np_features = dataset['features']
np_labels = dataset['labels']

#test the api
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"test": "test"})

#prediction with tensorflow
@app.route('/predict_tf', methods=['POST'])
def predict():
    data = request.get_json()
    vector = data['vector']
    vector = np.array(vector).reshape(1, -1)
    prediction = model.predict(vector)
    return jsonify({"prediction": prediction.tolist()})

#prediction with sklearn logarithmic regression
@app.route('/predict_scikit', methods=['POST'])
def predict_scikit():
    data = request.get_json()
    vector = data['vector']
    vector = np.array(vector).reshape(1, -1)
    model_2.fit(np_features, np_labels)
    prediction = model_2.predict(vector)
    return jsonify({"prediction": prediction.tolist()})

# view the sns paiplot of the dataset
@app.route('/plot', methods=['GET'])
def plot():
    sns.pairplot(data=dataset, hue='label')
    plt.savefig('plot.png')
    return jsonify({"plot": "plot.png"})

if __name__ == '__main__':
    app.run(debug=True)
    
    print("hello world")