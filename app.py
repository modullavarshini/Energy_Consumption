from flask import Flask, redirect, url_for, request, render_template
import pickle
import pandas as pd
import numpy as np
import hts
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import altair as alt
from io import BytesIO
import base64
warnings.simplefilter("ignore")

app = Flask(__name__)

# load the data
df = pd.read_csv("Natural Gas.csv")
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Date'] = df['Date'].dt.date
df.columns = [col_name.lower() for col_name in df.columns]
df_state_level = df.groupby(["date", "state"]).sum().reset_index(drop=False).pivot(index="date", columns="state", values="consumption")
df_total = df.groupby("date")["consumption"].sum().to_frame().rename(columns={"consumption": "total"})
# join the DataFrames
hierarchy_df = df_state_level.join(df_total)
hierarchy_df.index = pd.to_datetime(hierarchy_df.index)
hierarchy_df = hierarchy_df.resample("MS").sum()
loaded_model = pickle.load(open("fbprophet.pckl", "rb"))
pred_3 = loaded_model.predict(steps_ahead=3)
pred_4 = loaded_model.predict(steps_ahead=4)
pred_5 = loaded_model.predict(steps_ahead=5)
pred_6 = loaded_model.predict(steps_ahead=6)
pred_9 = loaded_model.predict(steps_ahead=9)
pred_10 = loaded_model.predict(steps_ahead=10)
pred_11 = loaded_model.predict(steps_ahead=11)
pred_12 = loaded_model.predict(steps_ahead=12)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        state=request.form['state']
        steps_ahead=int(request.form['shortterm'])
        if(steps_ahead==3):
            predictions_df = pred_3
        elif(steps_ahead==4):
            predictions_df = pred_4
        elif(steps_ahead==5):
            predictions_df = pred_5
        elif(steps_ahead==6):
            predictions_df = pred_6
        elif(steps_ahead==9):
            predictions_df = pred_9
        elif(steps_ahead==10):
            predictions_df = pred_10
        elif(steps_ahead==11):
            predictions_df = pred_11
        elif(steps_ahead==12):
            predictions_df = pred_12
        table_df = pd.DataFrame()
        table_df = predictions_df[[state]].copy()
        table_df = table_df.tail(steps_ahead)
        fig, ax = plt.subplots()
        predictions_df[state].plot(ax=ax, label="Predicted")
        hierarchy_df[state].plot(ax=ax, label="Observed")
        ax.legend()
        ax.set_title(state)
        ax.set_xlabel("Year")
        ax.set_ylabel("Consumption");
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return render_template('index.html',  tables=[table_df.to_html(classes='data')], titles=table_df.columns.values, plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
