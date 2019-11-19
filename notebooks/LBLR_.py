import pandas as pd
import numpy as np
%matplotlib inline
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

csvpath = Path('../data/cleandata/Data.csv')
Master_df = pd.read_csv(csvpath, parse_dates=True, index_col='Date')

def LRLB(Master_df):
    df = Master_df
    
    df['VLO_Return'] = df['VLO'].pct_change() * 100
    df['Brent_Return'] = df['Brent'].pct_change() * 100
    df['Lagged_Brent_Return'] = df.Brent_Return.shift()
    df = df.dropna()
    
    train = df['2017':'2018']
    test = df['2019']
    
    X_train = train["Lagged_Brent_Return"].to_frame()
    y_train = train["VLO_Return"]
    X_test = test["Lagged_Brent_Return"].to_frame()
    y_test = test["VLO_Return"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    out_of_sample_results = y_test.to_frame()
    
    out_of_sample_results["Out-of-Sample Predictions"] = model.predict(X_test)
    
    out_of_sample_mse = mean_squared_error(
    out_of_sample_results["VLO_Return"],
    out_of_sample_results["Out-of-Sample Predictions"])
    
    out_of_sample_rmse = np.sqrt(out_of_sample_mse)

    weeks = df.index.to_period("w").unique()

    training_window = 5
    timeframe = len(weeks) - training_window - 1

    all_predictions = pd.DataFrame(columns=["Out-of-Sample Predictions"])
    all_actuals = pd.DataFrame(columns=["Actual Returns"])
    
    for i in range(0, timeframe):    
        # Beginning of training window
        start_of_training_period = weeks[i].start_time.strftime(format="%Y-%m-%d")
    
        # End of training window
        end_of_training_period = weeks[training_window+i].end_time.strftime(format="%Y-%m-%d")

        # Window of test-window data
        test_week = weeks[training_window + i + 1]
    
        # String of testing window
        start_of_test_week  = test_week.start_time.strftime(format="%Y-%m-%d")
        end_of_test_week = test_week.end_time.strftime(format="%Y-%m-%d")
    
        train = df.loc[start_of_training_period:end_of_training_period]
        test = df.loc[start_of_test_week:end_of_test_week]
    
        # Create new dataframes:
        X_train = train["Lagged_Brent_Return"].to_frame()
        y_train = train["VLO_Return"]
        X_test = test["Lagged_Brent_Return"].to_frame()
        y_test = test["VLO_Return"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
        # Create a temporary dataframe to hold predictions
        predictions = pd.DataFrame(predictions, index=X_test.index, columns=["Out-of-Sample Predictions"])
    
        # Create a temporary DataFrame of the actual "y"s in the test dataframe, with column name="Actual Returns": 
        actuals = pd.DataFrame(y_test, index=y_test.index)
        actuals.columns = ["Actual Returns"]  
    
        # Append these two dataframes (predictions and actuals) to the two master DataFrames built outside the loop
        all_predictions = all_predictions.append(predictions)
        all_actuals = all_actuals.append(actuals) 
    Results = pd.concat([all_actuals, all_predictions], axis=1)

    results_2019 = Results.loc['2019':]
    
    mse = mean_squared_error(
    results_2019["Actual Returns"],
    results_2019["Out-of-Sample Predictions"])

    # Using that mean-squared-error, calculate the root-mean-squared error (RMSE):
    rolling_rmse = np.sqrt(mse)
    
    RMSE = print(f"Out-of-sample Root Mean Squared Error (RMSE): {out_of_sample_rmse}")
    RRMSE = print(f"Rolling Out-of-Sample Root Mean Squared Error (RMSE): {rolling_rmse}")
    
    
    results_2019 = results_2019
    results = results_2019.plot(subplots=True) 
    
    
    return RMSE, RRMSE, results_2019, results  

