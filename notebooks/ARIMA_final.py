from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
   
def arima_fuc (df):
    df1 = df[:-5]
    model = ARIMA(df1.VLO.values, order=(2, 1, 1))   
    results = model.fit()
    forecast = pd.DataFrame(results.forecast(steps=5)[0])
    
    forecast_df = forecast.rename(columns = {0 : "Predicted"})
    
    df2 = df['VLO'].tail(5)
    df2 = pd.DataFrame(df2)
    df2 = df2.reset_index(drop=True)
    df2 = df2.rename(columns = {"VLO" : "Actuals"})
    
    results = pd.concat([df2, forecast_df], join='outer', axis=1)
    
    return results