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

# Load the model.
model = load_model('model.h5')

# Load the logistic regression model.
model_2 = LogisticRegression()

# Load the dataset.
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
    # Read the input data.
    data = request.get_json()
    vector = data['vector']
    
    # Transform the input data into a numpy array.
    vector = np.array(vector).reshape(1, -1)
    
    # Use the model to make a prediction.
    prediction = model.predict(vector)
    
    # Return the prediction as a JSON object.
    return jsonify({"prediction": prediction.tolist()})

#prediction with sklearn logarithmic regression
# Add types to this code:
@app.route('/predict_scikit', methods=['POST'])
def predict_scikit():
    data: dict = request.get_json()
    vector: list = data['vector']
    vector: np.ndarray = np.array(vector).reshape(1, -1)
    model_2.fit(np_features, np_labels)
    prediction: list = model_2.predict(vector)
    return jsonify({"prediction": prediction.tolist()})

# view the sns paiplot of the dataset
@app.route('/plot', methods=['GET'])
def plot():
    sns.pairplot(data=dataset, hue='label')
    plt.savefig('plot.png')
    print('plot generated')
    return jsonify({"plot": "plot.png"})

if __name__ == '__main__':
    app.run(debug=True)