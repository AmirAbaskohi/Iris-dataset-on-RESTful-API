# Import libraries

from sklearn.preprocessing import Normalizer
from sklearn import metrics

import keras
import pandas as pd
import numpy as np

from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils

import pickle

# This function evaluate the model using sklearn evaluators
def evaluate(predicted, actual):
	print("Accuracy: ", metrics.accuracy_score(actual, predicted))
	print("Balanced Accuracy: ", metrics.balanced_accuracy_score(actual, predicted))
	print("Recall: ", metrics.recall_score(actual, predicted, average="weighted"))
	print("Precision: ", metrics.precision_score(actual, predicted, average="weighted"))
	print("F1: ", metrics.f1_score(actual, predicted, average="weighted"))

# Reading data from csv file and storing in dataframe
data = pd.read_csv("../data/iris.csv")

# Numerizing species feature
data.loc[data["variety"]=="Setosa","variety"] = 0
data.loc[data["variety"]=="Versicolor","variety"] = 1
data.loc[data["variety"]=="Virginica","variety"] = 2

# Printing first five elements
print(data.head())

# Suffling the dataset
data=data.iloc[np.random.permutation(len(data))]

# Makiing our features and traget dataframes
X=data.iloc[:,0:4].values
y=data.iloc[:,4].values

# Normalizing the data
normalizer = Normalizer()
X = normalizer.fit_transform(X)

# Train and test split
total_length = len(data)
train_length = int(0.8*total_length)
test_length = int(0.2*total_length)
X_train = X[:train_length]
X_test = X[train_length:]
y_train = y[:train_length]
y_test = y[train_length:]

# One hot encoding for target
y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)

# Creating the model
model=Sequential()
model.add(Dense(1000, input_dim=4, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Printing model summary
print(model.summary())

# Training model
model.fit(X_train,y_train,validation_data=(X_test, y_test),batch_size=20,epochs=10,verbose=1)

# Testing the dataset
prediction=model.predict(X_test)
prediction = np.argmax(prediction, axis=1)
y_test = np.argmax(y_test, axis=1)

# Evaluating the results
evaluate(prediction, y_test)

# Saving the model and normalizer
valid_inp = False
while not valid_inp:
	inp = input("Save the model? [Y/N]")
	if inp == 'Y' or inp == 'y':
		model.save('iris_classifier.h5', save_format='h5')
		with open('normalizer.pickle', 'wb') as f:
			pickle.dump(normalizer, f, protocol=pickle.HIGHEST_PROTOCOL)
		valid_inp = True
	elif inp == 'N' or inp == 'n':
		valid_inp = True
	else:
		print("Invalid input!")



