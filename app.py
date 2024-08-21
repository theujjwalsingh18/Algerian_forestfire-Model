import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('models/forst-ridge.pkl', 'rb'))
scaler = pickle.load(open('models/forst-scala.pkl', 'rb'))

@app.route('/',methods=['POST','GET'])
def predict():
    if request.method =='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        data = [[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]
        
        # Applying Stander Scaling to the get data from url
        data_scaled = scaler.transform(data)
        print(data_scaled)
        result = model.predict(data_scaled)
        print(result)
        
        return render_template('index.html',result=round(result[0],2))
        
    else:
        return render_template('index.html')  



if __name__=="__main__":
    app.run(host="0.0.0.0")