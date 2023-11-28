# %%
from icecream import ic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import tensorflow as tf

# %% [markdown]
# # Problematic of our dataset : revenues based upon the online shoppers intention

# %% [markdown]
# ### goal : to analyse the dataset, and be able to predict if a consumer is going to pay or not

# %%
dataset = pd.read_csv("online_shoppers_intention.csv")
dataset.head()

# %%
dataset.info()

# %% [markdown]
# # Data Processing

# %%
# imputation of null values
dataset.dropna(inplace=True) # no null values found
dataset.info()

# %%
#normalizing the dataset
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

dataset[['Administrative_Duration','Informational_Duration','ProductRelated_Duration']] = scaler.fit_transform(dataset[['Administrative_Duration','Informational_Duration','ProductRelated_Duration']])

# %%
dataset[["BounceRates", "ExitRates", "PageValues", "SpecialDay"]] = scaler.fit_transform(dataset[["BounceRates", "ExitRates", "PageValues", "SpecialDay"]])

# %%
dataset.head(100)

# %%
# replacing months with the conrresponding number 
dataset['Month'] = dataset['Month'].replace(['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'],[1,2,3,4,5,6,7,8,9,10,11,12])
dataset.head()

# %%
# categorization of VisitorType, weekend and Revenue
dataset['VisitorType'] = dataset['VisitorType'].replace(['Returning_Visitor','New_Visitor','Other'],[1,2,3])
dataset['Weekend'] = dataset['Weekend'].replace([True,False],[1,0])
dataset['Revenue'] = dataset['Revenue'].replace([True,False],[1,0])
dataset.info()

# %%
# dividing the dataset into features and labels

features = dataset.drop('Revenue', axis=1)
labels = dataset['Revenue']
labels = labels.replace([True, False], [1, 0])
labels = labels.astype('int')
np_labels = np.array(labels)
np_features = np.array(features)
np_features = np_features.astype('float')
features.info()

# %% [markdown]
# # Visualizing

# %%
# plotting different features against each other
plt.plot(dataset['Administrative_Duration'], dataset['Informational_Duration'], 'o', color='black');
plt.xlabel('Administrative_Duration')
plt.ylabel('Informational_Duration')
plt.show()

# %%
plt.plot(dataset['Administrative_Duration'], dataset['ProductRelated_Duration'], 'o', color='black');
plt.xlabel('Administrative_Duration')
plt.ylabel('ProductRelated_Duration')
plt.show()

# %%
# sort of a heatmap
sns.pairplot(dataset, hue='Revenue', vars=['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'])
plt.show()

# %%
sns.pairplot(dataset, hue='Revenue', vars=['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'], kind = "kde")
plt.show()

# %%
# PCA projection to 2D
from sklearn.decomposition import PCA
#create a dataset containing all the non categorical features
tmp_dataset = dataset[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Revenue']]
pca = PCA(n_components=2)
pca.fit(tmp_dataset)
X_pca = pca.transform(tmp_dataset)
print("Original shape: {}".format(str(tmp_dataset.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# plot first vs. second principal component, colored by class
plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=tmp_dataset['Revenue'], edgecolor='none', alpha=0.7, s=40)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

# %% [markdown]
# # Training

# %%
# we are going to use dbscan and kmeans clustering algorithms to search for clusters in the dataset
from sklearn.cluster import DBSCAN, KMeans

tmp_dataset = dataset.drop('Month', axis=1)
tmp_dataset = tmp_dataset.drop('VisitorType', axis=1)


dbscan = DBSCAN(eps=0.1, min_samples=5) # problem with the eps value
KMeans = KMeans(n_clusters=2)

dbscanned = dbscan.fit(tmp_dataset)
kmeaned = KMeans.fit(tmp_dataset)


# %%
tmp_dataset.info()

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(dataset)
# plot the dbscan clusters
plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeaned.labels_, edgecolor='none', alpha=0.7, s=40)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

# %%
np_labels.shape

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the parameter grid for Grid Search
param_grid = {'C': [i for i in np.arange(1, 10000,10)]}

# Create the Logistic Regression model
model = LogisticRegression(max_iter=10000)

# Create the Grid Search object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)

# Fit the Grid Search object to the data
grid_search.fit(np_features, np_labels)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)


# %%
best_params = {'C': 1121} # hard coded so that the grid search doesn't run every time
best_score = 0.880

# %%
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
plt.show()


# %% [markdown]
# #### Conclusion : a lot of values are missing, we need to find a way to fill them

# %% [markdown]
# # Bonus : label prediction with Tensorflow

# %% [markdown]
# **idea** : We can try to use a neural network to predict the labels of our dataset. We will use the Tensorflow library to do so.

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# split the dataset into train and test
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(np_features, np_labels, test_size=0.2, random_state=42)

# create the model

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[17]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train the model
history = model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=1, epochs=1000
)

# %%
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
# plot the loss and accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %%
# we save the model to use it later
model.save('model.h5')


