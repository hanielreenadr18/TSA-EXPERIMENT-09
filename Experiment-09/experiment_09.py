import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
sales = [100,105,103,110,115,112,120,125,123,130,135,132,
         140,145,143,150,155,152,160,165,163,170,175,172]
data = pd.Series(sales, name="Sales")
plt.figure(figsize=(10,4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month"); plt.ylabel("Sales")
plt.grid(True); plt.tight_layout(); plt.show()
adf_result = adfuller(data)
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))
diff_data = data.diff().dropna()
print(diff_data.to_string(index=False))
adf_diff = adfuller(diff_data)
print("Differenced ADF Statistic:", round(adf_diff[0], 4))
print("Differenced p-value:", round(adf_diff[1], 4))
plot_acf(diff_data, lags=10); plt.show()
plot_pacf(diff_data, lags=10, method="ywm"); plt.show()
model = ARIMA(data, order=(1,1,1))
fitted_model = model.fit()
print(fitted_model.summary())
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))
residuals = fitted_model.resid
plt.plot(residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Model Residuals")
plt.grid(True); plt.show()
plot_acf(residuals, lags=10); plt.show()
print(acorr_ljungbox(residuals, lags=[5], return_df=True))
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
for month, value in enumerate(forecast_mean, start=25):
    print(f"Month {month}: {value:.2f}")
print(forecast_ci.round(2))
future_months = list(range(len(data), len(data)+6))
plt.plot(range(len(data)), data.values, marker="o", label="Actual Sales")
plt.plot(future_months, forecast_mean.to_numpy(), marker="o",
         linestyle="--", label="Forecast")
plt.fill_between(future_months, forecast_ci.iloc[:,0].to_numpy(),
                 forecast_ci.iloc[:,1].to_numpy(), alpha=0.2,
                 label="95% Confidence Interval")
plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month"); plt.ylabel("Sales")
plt.legend(); plt.grid(True); plt.show()
