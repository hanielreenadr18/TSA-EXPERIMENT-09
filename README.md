# Experiment 09 – The Box–Jenkins Model (ARIMA)

### Name: HANIEL REENA D R
### Register No: 2305001008

## Aim
To understand and implement the Box–Jenkins approach to time-series forecasting using an ARIMA model.

## Algorithm
1. Import the required libraries.
2. Create monthly sales data and plot the time series.
3. Check stationarity using the ADF test.
4. Apply first differencing and test stationarity again.
5. Plot ACF and PACF.
6. Fit an ARIMA(1,1,1) model.
7. Display AIC and BIC values.
8. Check residuals and perform the Ljung–Box test.
9. Forecast the next six months and plot the forecast.

## Program
File: `experiment_09.py`

```python
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
```

## Output

### 1. Monthly Sales Data
<img width="960" height="295" alt="image" src="https://github.com/user-attachments/assets/85f444c6-b170-4188-b7c0-0ba7230bee4f" />


### 2. Autocorrelation (ACF)
<img width="430" height="280" alt="image" src="https://github.com/user-attachments/assets/11218a10-1d16-4898-8903-49ec93dcad68" />


### 3. Partial Autocorrelation (PACF)
<img width="475" height="280" alt="image" src="https://github.com/user-attachments/assets/084b5ba0-bbdd-43cb-8811-a0649aafa723" />


### 4. First-Differenced Sales
<img width="960" height="290" alt="image" src="https://github.com/user-attachments/assets/b2fd1fe5-56b1-49f4-a8f6-8bcfc13cc38b" />


### 5. Six-Month ARIMA Forecast
<img width="960" height="365" alt="image" src="https://github.com/user-attachments/assets/3f0c0601-e593-430c-bd25-9f16bb85a993" />


### 6. ARIMA Model Residuals
<img width="445" height="250" alt="image" src="https://github.com/user-attachments/assets/e199a5e7-ec43-4730-9afd-2e0e912f9bee" />


### 7. Residual Autocorrelation
<img width="460" height="250" alt="image" src="https://github.com/user-attachments/assets/51a12cb5-5563-4ef1-af27-929ebdac5294" />


## Observation
The original sales series shows an upward trend. The ADF test gives a p-value of 0.9253, indicating that the original series is non-stationary.
After first differencing, the ADF p-value is 0.0. The ARIMA(1,1,1) model was fitted and used to forecast the next six months. The Ljung–Box
test at lag 5 gives a p-value of 0.989959.

## Result
Thus, the Box–Jenkins ARIMA model was implemented, checked, and used to forecast six future sales values.
