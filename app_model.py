from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

root_path= "/home/Amaia/Taller_Despliegue_Directo/"

app = Flask(__name__)
app.config['DEBUG'] = True

# Enruta la landing page (endpoint /)

@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    #return "Bienvenido a mi genial API del modelo advertising que inventé yo cuando tenía 5 años"
    return '''
      <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Revenue Prediction Model</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            text-align: center;
            padding: 20px;
            background: #fff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }
        h1 {
            color: #5a67d8;
        }
        p {
            font-size: 1.2em;
        }
        .btn {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 1em;
            color: #fff;
            background-color: #5a67d8;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #434190;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to the Revenue Prediction Model</h1>
        <p>
            This platform allows you to predict your revenue based on your marketing investment.
            Simply provide the necessary input data and let our model analyze and generate
            accurate predictions to optimize your strategy.
        </p>
        <p>
            Ready to get started? Click the button below to explore our prediction tools.
        </p>
        <a href="/predict" class="btn">Get Started</a>
    </div>
</body>
</html>
'''
# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods= ['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    model = pickle.load(open(root_path+'ad_model.pkl', 'rb'))
    #model = pickle.load(open('ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])

    return jsonify({'predictions': prediction[0]})

# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods= ['GET'])
def retrain(): # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv(root_path + 'data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        pickle.dump(model, open(root_path + 'ad_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__== "__main__":
    app.run()