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
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(scaled_data)
    print(output[0])
    return jsonify({'output': output[0]})


# TODO: Write a api to get data from form
@app.route("/predict",methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text = f'The predictied house price is: {output}')

if __name__ == '__main__':
    app.run(debug = True)
    