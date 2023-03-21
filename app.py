from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict_placement():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scal_trans=scaler.transform(final_features)
    prediction = model.predict(scal_trans)

    output = prediction[0]
    if output==1:
        return render_template("index3.html")
    else:
        return render_template("index2.html")     
if __name__ == '__main__':
    app.run(port=8080)