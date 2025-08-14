from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import dask.dataframe as dd
import pandas as pd

app = Flask(__name__)
Bootstrap(app)

def load_weather():
    df = dd.read_csv("weather_2025_cleaned.csv")
    df["DATE"] = dd.to_datetime(df["DATE"], errors='coerce')
    df = df.dropna(subset=["DATE"]).compute()
    df.columns = [col.lower() for col in df.columns]

    stations_info = pd.read_csv("vietnam_stations.csv")
    stations_info.columns = [col.lower() for col in stations_info.columns]

    df["date"] = pd.to_datetime(df["date"])
    df["day_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour

    df = df.merge(stations_info, left_on="station", right_on="station_id", how="left")
    return df

def classify_weather(row):

    if row["dew_c"] > 23 and row["cloud_ceiling"] > 4000 and row["vis_m"] < 5000:
        return "rainy"
    else:
        return "sunny"

@app.route("/")
def index():
    df = load_weather()

    # Bộ lọc
    input_date = request.args.get("date")
    input_station = request.args.get("station_id", "").strip()
    latest_date = df["day_only"].max() if not input_date else pd.to_datetime(input_date).date()
    df_filtered = df[df["day_only"] == latest_date]

    if input_station:
        df_filtered = df_filtered[df_filtered["station_id"] == input_station]

    # Dự báo 5 ngày
    forecast = (
        df.groupby("day_only")[["tmp_c"]]
        .agg(["max", "min"])
        .reset_index()
        .sort_values(by="day_only")
        .head(5)
    )
    forecast.columns = ["date", "max_temp", "min_temp"]
    forecast["day"] = pd.to_datetime(forecast["date"]).dt.strftime("%A")

    # Tính trung bình theo ngày để phân loại thời tiết
    daily = df.resample("D", on="date").agg({
        "dew_c": "mean",
        "cloud_ceiling": "mean",
        "vis_m": "mean"
    }).reset_index()

    daily["weather_type"] = daily.apply(classify_weather, axis=1)

    # Tính số ngày nắng và mưa
    weather_counts = daily["weather_type"].value_counts()
    sunny_days = int(weather_counts.get("sunny", 0))
    rainy_days = int(weather_counts.get("rainy", 0))
    avg_high = round(df.resample("D", on="date")["tmp_c"].max().mean(), 1)
    avg_low = round(df.resample("D", on="date")["tmp_c"].min().mean(), 1)

    # Dữ liệu cho biểu đồ xu hướng
    trend_df = (
        df.resample("D", on="date")["tmp_c"].agg(["max", "min"]).reset_index()
    )
    trend_data = {
        "labels": trend_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "highs": trend_df["max"].round(1).tolist(),
        "lows": trend_df["min"].round(1).tolist()
    }

    # Dự báo hàng giờ
    selected_hours = [6, 9, 12, 15, 18]
    hourly = (
        df_filtered[df_filtered["hour"].isin(selected_hours)]
        .groupby("hour")[["tmp_c", "dew_c"]]
        .mean()
        .round(1)
        .reset_index()
    )

    latest = df_filtered.sort_values(by="date", ascending=False).iloc[0] if not df_filtered.empty else df.iloc[0]

    # Dữ liệu trạm
    stations = df[['station_name', 'latitude', 'longitude', 'tmp_c', 'wind_speed']]\
        .dropna(subset=["latitude", "longitude"])\
        .drop_duplicates(subset='station_name')\
        .head(100)
    station_data = stations.to_dict(orient='records')

    # Thông tin khí hậu (tháng)
    monthly = df.groupby(df["date"].dt.month).agg({
        "tmp_c": "mean",
        "dew_c": "mean",
        "wind_speed": "mean"
    }).reset_index()

    hot_month = int(monthly.loc[monthly["tmp_c"].idxmax(), "date"])
    cold_month = int(monthly.loc[monthly["tmp_c"].idxmin(), "date"])
    wet_month = int(monthly.loc[monthly["dew_c"].idxmax(), "date"])
    wind_month = int(monthly.loc[monthly["wind_speed"].idxmax(), "date"])

    # Tóm tắt 12 tháng trước
    temp_high = df.resample("D", on="date")["tmp_c"].max()
    temp_low = df.resample("D", on="date")["tmp_c"].min()
    wind_daily = df.resample("D", on="date")["wind_speed"].mean()
    rain_daily = df["precip_mm"].resample("D", on="date").sum() if "precip_mm" in df else None

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
            "max": round(rain_daily.max()/10, 2),  
            "avg": round(rain_daily.mean()/10, 2),
            "min": round(rain_daily.min()/10, 2)
        },
        "wind": {
            "max": round(wind_daily.max(), 1),
            "avg": round(wind_daily.mean(), 1),
            "min": round(wind_daily.min(), 1)
        }
    }
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
        forecast=forecast.to_dict(orient="records"),
        hourly=hourly.to_dict(orient="records"),
        stations=station_data,
        hot_month=hot_month,
        cold_month=cold_month,
        wet_month=wet_month,
        wind_month=wind_month,
        climate_summary=climate_summary,
        sunny_days=sunny_days,
        rainy_days=rainy_days,
        trend_data=trend_data
    )

if __name__ == "__main__":
    app.run(debug=True)
