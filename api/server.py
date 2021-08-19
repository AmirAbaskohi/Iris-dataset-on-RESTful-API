from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
from tensorflow import keras
import os
import pickle

app = Flask(__name__)
api = Api(app)

model = keras.models.load_model("../model/iris_classifier.h5")
data = pd.read_csv("../data/iris.csv")

classes = ['Setosa', 'Versicolor', 'Virginica']

with open('../model/normalizer.pickle', 'rb') as f:
    normalizer = pickle.load(f)

class Classifier(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('sepal_length', required=True)
        parser.add_argument('sepal_width', required=True)
        parser.add_argument('petal_length', required=True)
        parser.add_argument('petal_width', required=True)
        args = parser.parse_args()

        sepal_length = float(args["sepal_length"])
        sepal_width = float(args["sepal_width"])
        petal_length = float(args["petal_length"])
        petal_width = float(args["petal_width"])

        values = [[sepal_length, sepal_width, petal_length, petal_width]]

        values = normalizer.transform(values)

        return {'result': classes[np.argmax(model.predict(np.array(values)), axis=1)[0]]}, 200

api.add_resource(Classifier, '/classifier')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))