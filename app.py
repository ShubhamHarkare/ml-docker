import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas 

app = Flask(__name__)

# TODO: Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))

#TODO:  Load the scaler
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/predict_api',methods = ['GET','POST'])
def predict_api():
    data = request.json['data']
    # print(data.values())
    # data = np.array(list(data.values)).reshape(1,-1)
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(scaled_data)
    print(output[0])
    return jsonify({'output': output[0]})
    # return jsonify(output)


if __name__ == '__main__':
    app.run(debug = True)
    