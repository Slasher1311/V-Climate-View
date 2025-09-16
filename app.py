from flask import Flask, render_template, request
import pandas as pd
from pmdarima import auto_arima
from datetime import timedelta

app = Flask(__name__)

def load_weather():
    df = pd.read_csv("weather_2025_cleaned.csv", parse_dates=["DATE"])
    df.columns = [col.lower() for col in df.columns]

    stations_info = pd.read_csv("vietnam_stations.csv")
    stations_info.columns = [col.lower() for col in stations_info.columns]

    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df["day_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour

    df = df.merge(stations_info, left_on="station", right_on="station_id", how="left")
    return df

def forecast_future_arima_all(df, days=7):
    daily = df.resample("D", on="date")["tmp_c"].agg(["mean"]).dropna().reset_index()
    y_mean = daily["mean"].values
    model_mean = auto_arima(y_mean, seasonal=True, m=7, trace=False, error_action="ignore", suppress_warnings=True)
    forecast_mean = model_mean.predict(n_periods=days)

    future_dates = pd.date_range(daily["date"].iloc[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    fc = pd.DataFrame({
        "ds": future_dates,
        "mean_temp": forecast_mean.round(1),
        "day": future_dates.strftime("%A")
    })
    return fc

@app.route("/")
def index():
    df = load_weather()
    range_days = request.args.get("range", default="365")
    try:
        range_days = int(range_days)
    except:
        range_days = 365

    max_date = df["date"].max()
    min_date = max_date - timedelta(days=range_days)

    df_filtered_for_trend = df[(df["date"] >= min_date) & (df["date"] <= max_date)]

    input_date = request.args.get("date")
    input_station = request.args.get("station_id", "").strip()
    input_station_name = request.args.get("station_name", "").strip().lower()

    latest_date = df["day_only"].max() if not input_date else pd.to_datetime(input_date).date()
    df_filtered = df[df["day_only"] == latest_date]

    if input_station:
        df_filtered = df_filtered[df_filtered["station_id"] == input_station]
    if input_station_name:
        df_filtered = df_filtered[
            df_filtered["station_name"].str.lower().str.contains(input_station_name, na=False)
        ]

    forecast_7 = forecast_future_arima_all(df, days=7)
    forecast_14 = forecast_future_arima_all(df, days=14)
    forecast_21 = forecast_future_arima_all(df, days=21)

    daily_trend = df_filtered_for_trend.resample("D", on="date")["tmp_c"].agg(["max", "min", "mean"]).dropna().reset_index()

    trend_data = {
        "labels": daily_trend["date"].dt.strftime("%Y-%m-%d").tolist(),
        "highs": daily_trend["max"].round(1).tolist(),
        "lows": daily_trend["min"].round(1).tolist(),
        "means": daily_trend["mean"].round(1).tolist()
    }

    selected_hours = [3, 6, 9, 12, 15, 18, 21, 24]
    hourly = (
        df_filtered[df_filtered["hour"].isin(selected_hours)]
        .groupby("hour")[["tmp_c", "dew_c"]]
        .mean()
        .round(1)
        .reset_index()
    )

    latest = df_filtered.sort_values(by="date", ascending=False).iloc[0] if not df_filtered.empty else df.iloc[0]

    stations = df[['station_name', 'latitude', 'longitude', 'tmp_c', 'wind_speed']] \
        .dropna(subset=["latitude", "longitude"]) \
        .drop_duplicates(subset='station_name') \
        .head(100)
    station_data = stations.to_dict(orient='records')

    monthly = df.groupby(df["date"].dt.month).agg({
        "tmp_c": "mean",
        "dew_c": "mean",
        "wind_speed": "mean"
    }).reset_index()

    hot_month = int(monthly.loc[monthly["tmp_c"].idxmax(), "date"])
    cold_month = int(monthly.loc[monthly["tmp_c"].idxmin(), "date"])
    wet_month = int(monthly.loc[monthly["dew_c"].idxmax(), "date"])
    wind_month = int(monthly.loc[monthly["wind_speed"].idxmax(), "date"])

    temp_high = df.resample("D", on="date")["tmp_c"].max()
    temp_low = df.resample("D", on="date")["tmp_c"].min()
    wind_daily = df.resample("D", on="date")["wind_speed"].mean()
    rain_daily = df["precip_mm"].resample("D", on="date").sum() if "precip_mm" in df.columns else None

    climate_summary = {
        "high_temp": {
            "max": round(temp_high.max(), 1),
            "avg": round(temp_high.mean(), 1),
            "min": round(temp_high.min(), 1)
        },
        "low_temp": {
            "max": round(temp_low.max(), 1),
            "avg": round(temp_low.mean(), 1),
            "min": round(temp_low.min(), 1)
        },
        "rain": None if rain_daily is None else {
            "max": round(rain_daily.max() / 10, 2),
            "avg": round(rain_daily.mean() / 10, 2),
            "min": round(rain_daily.min() / 10, 2)
        },
        "wind": {
            "max": round(wind_daily.max(), 1),
            "avg": round(wind_daily.mean(), 1),
            "min": round(wind_daily.min(), 1)
        }
    }

    avg_high = round(daily_trend["max"].mean(), 1) if not daily_trend.empty else None
    avg_low = round(daily_trend["min"].mean(), 1) if not daily_trend.empty else None

    return render_template(
        "index.html",
        latest_weather={
            "temp": latest.get("tmp_c", "N/A"),
            "feels_like": round(latest.get("tmp_c", 0) + 3, 1),
            "humidity": latest.get("dew_c", "N/A"),
            "wind": latest.get("wind_speed", "N/A"),
            "visibility": latest.get("vis_m", "N/A"),
            "pressure": latest.get("slp_hpa", "N/A"),
            "condition": "Nhiều mây"
        },
        forecast_7=forecast_7.to_dict(orient="records"),
        forecast_14=forecast_14.to_dict(orient="records"),
        forecast_21=forecast_21.to_dict(orient="records"),
        hourly=hourly.to_dict(orient="records"),
        stations=station_data,
        hot_month=hot_month,
        cold_month=cold_month,
        wet_month=wet_month,
        wind_month=wind_month,
        avg_high=avg_high,
        avg_low=avg_low,
        climate_summary=climate_summary,
        trend_data=trend_data,
        station_name=input_station_name,
        selected_range=range_days  
    )

if __name__ == "__main__":
    app.run(debug=True)
