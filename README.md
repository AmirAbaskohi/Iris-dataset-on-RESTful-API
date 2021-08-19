# RESTful API in datascience
This is an introduction to implementing RESTful APIs using python Flask. A basic model for Iris dataset, which is a very famous dataset, will be trained and used here as well.

## Introduction
A few software development talents function as floodgates, keeping a tremendous ocean of possibility at bay.
When you learn one of these talents, it's as if a whole new universe opens up in front of you. 

Here I am using one these skills. An API is the ability to build code that can be deployed online and communicated with. 

Building a deployable API requires a number of sub-skills, which we will go through as we put our API together.
Nonetheless, Rome wasn't built in a day, and neither is mastery (or even proficiency), so this isn't a five-minute guide to becoming a hero. 

Here I will:
* Build an API with `Flask`
* Package my app with `Docker`

## Requirments
* flask
* flask-restful
* tensorflow
* numpy
* pandas
* sklearn
* docker

## How to run?
First you can train your model. This can be done by running below command in `model` directory:
```
python main.py
```

Then run the flask server by running below command in `api` directory:
```
python server.py
```

Now server is accessible in `0.0.0.0:8080` port.

You can run the apis using postman. Here is an example:
![1](https://user-images.githubusercontent.com/50926437/130132648-1e798526-cf90-4b64-a0d1-0e57a936e900.png)


*Amirhossein Abaskohi*
