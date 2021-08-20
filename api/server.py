from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
from tensorflow import keras
import os
import pickle

# Creating flas application and API system
app = Flask(__name__)
api = Api(app)

# Importing our keras classfier for iris dataset
model = keras.models.load_model("../model/iris_classifier.h5")

# Importing the dataset
data = pd.read_csv("../data/iris.csv")

# Classes to return the result of our neural network
classes = ['Setosa', 'Versicolor', 'Virginica']

# Importing our normalizer to normalize our data
with open('../model/normalizer.pickle', 'rb') as f:
    normalizer = pickle.load(f)

# Adding a resource for classifing
class Classifier(Resource):
    # GET method
    def get(self):
        # Highlighting our parameters
        parser = reqparse.RequestParser()
        parser.add_argument('sepal_length', required=True)
        parser.add_argument('sepal_width', required=True)
        parser.add_argument('petal_length', required=True)
        parser.add_argument('petal_width', required=True)
        args = parser.parse_args()

        # Casting the parameters from string to float
        sepal_length = float(args["sepal_length"])
        sepal_width = float(args["sepal_width"])
        petal_length = float(args["petal_length"])
        petal_width = float(args["petal_width"])

        # Making a list of our parameters in a format which is acceptable for normalizer
        values = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Normalizing the data
        values = normalizer.transform(values)

        # Predicting the class and return the result
        return {'result': classes[np.argmax(model.predict(np.array(values)), axis=1)[0]]}, 200

# Adding a resource for data manipulation
class Data(Resource):
    # GET method to receive the dataset
    def get(self):
        return {'data': data.to_dict()}

    # POST method to add value to dataset
    def post(self):

        # Using global dataframe
        global data

        # Highlighting our parameters
        parser = reqparse.RequestParser()
        parser.add_argument('sepal_length', required=True)
        parser.add_argument('sepal_width', required=True)
        parser.add_argument('petal_length', required=True)
        parser.add_argument('petal_width', required=True)
        args = parser.parse_args()
        
        # Casting the parameters from string to float
        sepal_length = float(args["sepal_length"])
        sepal_width = float(args["sepal_width"])
        petal_length = float(args["petal_length"])
        petal_width = float(args["petal_width"])

        # Adding value to dataframe
        data = data.append({"sepal.length": sepal_length, "sepal.width": sepal_width,
                             "petal.length": petal_length, "petal.width":petal_width}, ignore_index=True)

        # Returning the result
        return {'result': 'New data added successfully'}, 200

    # PUT method to update a value in a dataset
    def put(self):

        # Using global dataframe
        global data

        # Highlighting our parameters
        parser = reqparse.RequestParser()
        parser.add_argument('index', required=True)
        parser.add_argument('sepal_length', required=True)
        parser.add_argument('sepal_width', required=True)
        parser.add_argument('petal_length', required=True)
        parser.add_argument('petal_width', required=True)
        args = parser.parse_args()

        # Casting the parameters from string to float
        index = int(args["index"])
        sepal_length = float(args["sepal_length"])
        sepal_width = float(args["sepal_width"])
        petal_length = float(args["petal_length"])
        petal_width = float(args["petal_width"])
        args = parser.parse_args()
    
        # Check if data want to change exists
        if index in data.index:
            # Updating data
            data.loc[index, "sepal.length"] = sepal_length
            data.loc[index, "sepal.width"] = sepal_width
            data.loc[index, "petal.length"] = petal_length
            data.loc[index, "petal.width"] = petal_width
            # Return the result
            return {'result': 'Data updated successfully'}, 200
        else:
            # Return data not found
            return {'result': 'Index not found'}, 400

    # DELETE method to delete a value from dataset
    def delete(self):
        
        # Using global dataframe
        global data

        # Highlighting our parameters
        parser = reqparse.RequestParser()
        parser.add_argument('index', required=True)
        args = parser.parse_args()

        # Casting the parameters from string to float
        index = int(args["index"])
        
        # Check if data want to delete exists
        if index in data.index:
            # Deleting data
            data = data.drop(index)
            # Return the result
            return {'result': 'Data deleted successfully'}, 200
        else:
            # Return data not found
            return {'result': 'Index not found'}, 400

# Adding resources
api.add_resource(Classifier, '/classifier')
api.add_resource(Data, '/data')

if __name__ == '__main__':
    # Running our server on 0.0.0.0 and port 8080
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))