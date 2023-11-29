import io
import flask
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import seaborn as sns

app = Flask(__name__)

# Tensorflow model
model = load_model('model.h5')

# Load the logistic regression model.
model_2 = LogisticRegression()
best_params = {'C': 1121}

#DBSCAN model
model_3 = DBSCAN()

# Load the dataset.
dataset = np.load('dataset.npz')
np_features = dataset['features']
np_labels = dataset['labels']
dataset = pd.DataFrame(np.hstack((np_features, np_labels.reshape(-1, 1))), columns = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
       'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',
       'Weekend', 'Revenue'])

#test the api
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"test": "test"})

#prediction with tensorflow
@app.route('/tf', methods=['GET'])
def predict():
    # Perform PCA on the dataset
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np_features)

    # Predict the labels using the best model
    predicted_labels = [1 if i > 0.5 else 0 for i in model.predict(np_features)]

    # Get the indices of the points that are misclassified
    misclassified_indices = np.where(predicted_labels != np_labels)[0]

    # Plot the misclassified points using PCA in 2 dimensions
    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np_labels, edgecolor='none', alpha=0.7, s=40)
    plt.scatter(X_pca[misclassified_indices, 0], X_pca[misclassified_indices, 1], c='red', edgecolor='none', alpha=0.7, s=40)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig('plot_tf.png')
    print('plot generated')
    return jsonify({"message" : "plotting complete, check the following path to see it","path": "plot_tf.png"})


@app.route('/scikit-kmeans', methods=['GET'])
def predict_scikit_kmeans():
    KMean = KMeans(n_clusters=2)
    kmeaned = KMean.fit(dataset)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(dataset)
    # plot the 2 clusters
    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeaned.labels_, edgecolor='none', alpha=0.7, s=40)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.savefig('plot_scikit-kmeans.png')
    print('plot generated')
    return jsonify({"message" : "plotting complete, check the following path to see it","path": "plot_scikit-kmeans.png"})


@app.route('/scikit-log', methods=['GET'])
def sci_predict_log():
    # Fit the Logistic Regression model with the best parameter
    best_model = LogisticRegression(max_iter=10000, C=best_params['C'])
    best_model.fit(np_features, np_labels)

    # Perform PCA on the dataset
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np_features)

    # Predict the labels using the best model
    predicted_labels = best_model.predict(np_features)

    # Get the indices of the points that are misclassified
    misclassified_indices = np.where(predicted_labels != np_labels)[0]

    # Plot the misclassified points using PCA in 2 dimensions
    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np_labels, edgecolor='none', alpha=0.7, s=40)
    plt.scatter(X_pca[misclassified_indices, 0], X_pca[misclassified_indices, 1], c='red', edgecolor='none', alpha=0.7, s=40)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig('plot_scikit-log.png')
    print('plot generated')
    return jsonify({"message" : "plotting complete, check the following path to see it","path": "plot_scikit-log.png"})


@app.route('/plot', methods=['GET'])
def plot():
    sns.pairplot(data=dataset, hue='Revenue')
    plt.savefig('sns_pair.png')
    print('plot generated')
    return jsonify({"message" : "plotting complete, check the following path to see it","path": "sns_pair.png"})

@app.route('/default', methods=['GET'])
def default():
    X_pca = PCA(n_components=2).fit_transform(dataset)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np_labels , edgecolor='none', alpha=0.7, s=40)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.savefig('plot_default.png')
    print('plot generated')
    return jsonify({"message" : "plotting complete, check the following path to see it","path": "plot_default.png"})
    
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the API!", "routes": "/test, /tf, /scikit-kmeans, /scikit-log, /plot, /default",
                    "/test": "test the api",
                    "/tf": "plot in 2D with PCA and tensorflow model. Red points are misclassified",
                    "/scikit-kmeans": "plot in 2D with PCA and scikit-learn kmeans",
                    "/scikit-log": "plot in 2D with PCA and scikit-learn logistic regression. Red points are misclassified",
                    "/plot": "sns pairplot, to see the correlation between the features",
                    "/default": "plot in 2D with PCA"
                    })


if __name__ == '__main__':
    app.run(debug=True)