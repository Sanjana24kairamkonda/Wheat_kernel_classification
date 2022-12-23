import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

wheat_classification=Flask(__name__)

model=pickle.load(open("model.pkl","rb"))

@wheat_classification.route("/")
def Home():
    return render_template("index.html")

@wheat_classification.route("/predict",methods= ["POST"])
def predict():
    float_features= [float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction= model.predict(features)

    return render_template("index.html", prediction_text = "wheat classification: {}".format(prediction))

if __name__ == "__main__":
    wheat_classification.run(debug=True)