from flask import Flask, render_template, request
import pickle
import pandas as pd


model = None
with open(r'C:\Users\PC\Desktop\project3\flask_app\pipe.pkl','rb') as pickle_file:
    model = pickle.load(pickle_file)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        return render_template('index.html'), 200


    if request.method == 'POST':

        def predict_price(model, brand, color, engine, gear, fuel, route, year, mileage):
            df = pd.DataFrame(
                data=[[brand, color, engine, gear, fuel, route, year, mileage]], 
                columns=['brand', 'color', 'engine', 'gear', 'fuel', 'route', 'year', 'mileage']
                )
            pred = model.predict(df)[0]

            return pred

        data1 = request.form['brand']
        data2 = request.form['color']
        data3 = request.form['engine']
        data4 = request.form['gear']
        data5 = request.form['fuel']
        data6 = request.form['route']
        data7 = request.form['year']
        data8 = request.form['mileage']

        result = predict_price(model, data1, data2, data3, data4, data5, data6, data7, data8)

        return render_template('index.html', result=result)

if __name__=="__main__":
    app.run()

