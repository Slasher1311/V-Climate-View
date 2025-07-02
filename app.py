from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import dask.dataframe as dd
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
Bootstrap(app)

weather_df = dd.read_csv("weather_cleaned.csv")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
