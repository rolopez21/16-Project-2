from statsmodels.tsa.arima_model import ARIMA
    
def arima_fuc (df):
    
    model = ARIMA(df.VLO.values, order=(2, 1, 1))   
    results = model.fit()
    forecast = pd.DataFrame(results.forecast(steps=5)[0])
    return results.summary(), forecast.plot(title="Futures Forecast")
