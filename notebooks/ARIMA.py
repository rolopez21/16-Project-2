from statsmodels.tsa.arima_model import ARIMA
    
def arima_fuc (df):
    
    model = ARIMA(df.VLO.values, order=(2, 1, 1))   
    results = model.fit()
    return (results)
