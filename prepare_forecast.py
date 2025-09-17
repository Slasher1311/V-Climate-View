import pandas as pd
from pmdarima import auto_arima

def load_weather_data():
    df = pd.read_csv("weather_2025_cleaned.csv", parse_dates=["DATE"])
    df.columns = [col.lower() for col in df.columns]
    df = df.dropna(subset=["date", "tmp_c"])
    df["date"] = pd.to_datetime(df["date"])
    return df

def forecast_future_arima(df, days=7):
    daily = df.resample("D", on="date")["tmp_c"].mean().dropna().reset_index()
    y = daily["tmp_c"].values

    model = auto_arima(
        y,
        seasonal=True,
        m=7,
        suppress_warnings=True,
        error_action="ignore"
    )

    forecast = model.predict(n_periods=days)
    future_dates = pd.date_range(start=daily["date"].iloc[-1] + pd.Timedelta(days=1), periods=days)

    result = pd.DataFrame({
        "ds": future_dates,
        "mean_temp": forecast.round(1),
        "day": future_dates.strftime("%A")
    })

    return result

def main():
    df = load_weather_data()

    forecast_7 = forecast_future_arima(df, days=7)
    forecast_30 = forecast_future_arima(df, days=30)

    forecast_7.to_csv("forecast_7.csv", index=False)
    forecast_30.to_csv("forecast_30.csv", index=False)


    print("✅ Dự báo đã lưu thành công vào forecast_7.csv, forecast_30.csv")

if __name__ == "__main__":
    main()
