import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("weather_cleaned.csv")

df = df.dropna(subset=['date', 'tmp_c'])
df['ds'] = pd.to_datetime(df['date'])  
df['y'] = df['tmp_c']
df = df.sort_values("ds")

model = Prophet()
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=30)  
forecast = model.predict(future)

forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print(forecast_result.tail())

fig1 = model.plot(forecast)
plt.title("Dự báo nhiệt độ 30 ngày tới")
plt.xlabel("Ngày")
plt.ylabel("Nhiệt độ (°C)")
plt.tight_layout()
plt.show()
