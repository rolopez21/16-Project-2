import pandas as pd
from datetime import datetime
from iexfinance.stocks import get_historical_data
import quandl

import bokeh
import numpy as np
import hvplot.pandas
import panel as pn

from LBLR3 import LRLB
from ARIMA import arima_fuc
from LSTM import LSTM_F

pn.extension()


def get_Quandl_Close (api, col, col_rename, start, end):
# Returns a dataframe with settle or close values for the given api & ticker combination

    df = quandl.get(api, start_date = start, end_date = end)
    df.rename(columns={col: col_rename}, inplace=True)
    return df.loc[ : , [col_rename]]



def get_Features(feature_list, start_date, end_date):
    #Return the settle values for multiple features as list of dataframes

    result_dfs = []
    
    for feature in feature_list:
        result_df = get_Quandl_Close(feature['api'], feature['close'], feature['rename'], start_date, end_date)
        result_dfs.append(result_df)
        
    return pd.concat(result_dfs, axis=1, join='inner')



def get_IEX_Close (ticker, start, end):
    # Returns a dataframe with close values for the given ticker and date range
     
    df = get_historical_data(ticker, start, end, output_format='pandas')
    
    #Rename the close column to ticker
    df.rename(columns={'close': ticker}, inplace=True)
    
    return df.loc[ : , [ticker]]



def create_Master_DF( ):
    quandl.ApiConfig.api_key = 'DKz2PtZF4s2ArXG4caAt'
    
    #Create a list of features, each feature is stored as dictionary
    feature_list = [
        {'api':'CHRIS/CME_CL1', 'close':'Settle', 'rename':'Crude1'},
        {'api':'CHRIS/CME_AFF1','close':'Settle', 'rename':'WTI_Midland'},
        {'api':'CHRIS/ICE_B1','close':'Settle', 'rename':'Brent'},
        {'api':'CHRIS/CME_RB1','close':'Settle', 'rename':'RBOB'},
        {'api':'CHRIS/CME_EH1','close':'Settle', 'rename':'Ethanol'},
        {'api':'CHRIS/CME_C1','close':'Settle', 'rename':'Corn'},
        {'api':'CHRIS/CME_WCC1','close':'Settle', 'rename':'CAD_Heavy'},
        {'api':'CHRIS/CBOE_VX1','close':'Settle', 'rename':'VIX'},
        {'api':'CHRIS/CME_HO1','close':'Settle', 'rename':'ULSD'},
        {'api':'OPEC/ORB','close':'Value', 'rename':'OPEC_Basket'},
        {'api':'CHRIS/CME_RM1','close':'Settle', 'rename':'RBOB_Crack'},
        {'api':'CHRIS/CME_GY1','close':'Settle', 'rename':'GC_ULSD_Crack'},
        {'api':'CHRIS/CME_HK1','close':'Settle', 'rename':'NY_ULSD_Crack'},
        {'api':'CHRIS/CME_NG1','close':'Settle', 'rename':'NG_HB'},
        {'api':'CHRIS/CME_SP1','close':'Settle', 'rename':'S_P'}             
    ]
    
    #Create Date Range
    end_date = '2019-06-30' # The date - 2019 Jun 30
    start_date = '2017-01-01' # The date - 2017 Jan 01
    
    #Convert to Datetime
    format_str = '%Y-%m-%d' # The format
    end_datetime = datetime.strptime(end_date, format_str)
    start_datetime = datetime.strptime(start_date, format_str)

    features = get_Features(feature_list, start_date, end_date)
    stock = get_IEX_Close('VLO', start_datetime, end_datetime)

    return pd.concat([features, stock], axis=1, join='inner')


def build_dashboard(master_df, corr_df, model1_df, model2_df, model3_df):
    """Build the dashboard with 4 tabs - intro, plus one of each model"""
    
    # Create tab_1_columns for Model 1
    tab_intro = pn.Column(
        ("# Team: Oil Futures"),
        ("## Team Members: Josue Gavarrete, Roger Lopez, Maria Rosa, Vinay Kakuru"),
        ("## Coorelation between Features and VLO Stock"),
        corr_df.hvplot.heatmap(x='stock', y='Features', C='VLO'),
        ("## Model 1 Dataset: Actual Returns vs. Predicted Returns."),
        master_df.head().hvplot.table(
            title="Master Data: Features and Stock VLO",
            columns=["Crude1","WTI_Midland","Brent","RBOB","Ethanol",
            "Corn","CAD_Heavy","VIX","ULSD","OPEC_Basket",
            "RBOB_Crack","GC_ULSD_Crack","NY_ULSD_Crack","NG_HB","S_P","VLO"], width=400
        )
    )
    
    # Create tab_1_columns for Model 1
    tab_1_column = pn.Column(
        ("# Model 1: Linear Regression on Lagged Brent"),
        ("## Model 1 Prediction vs. Actual"),
        model1_df.hvplot.line(y="Actual Returns", label="Act. Return")
        * model1_df.hvplot.line(y="Out-of-Sample Predictions", label="Pred. Return"),
        ("## Model 1 Dataset: Actual Returns vs. Predicted Returns."),
        model1_df.hvplot.table(
            title="Model 1: Predictions and Actual",
            columns=["Actual Returns", "Out-of-Sample Predictions"]
        )
    )

    # Create tab_2_columns for Model 2
    tab_2_column = pn.Column(
        ("# Model 2: ARIMA 5 Day Projection"),
        ("## Model 2 Prediction vs. Actual"),
        model2_df.hvplot.line(y="Actuals", label="Actual")
        * model2_df.hvplot.line(y="Predicted", label="Predicted"),
        ("## Model 2 Dataset"),
        model2_df.hvplot.table(
            title="Model 2 Predictions and Actual",
            columns=["Actuals", "Predicted"]
        )
    )

    # Create tab_3_columns for Model 3
    tab_3_column = pn.Column(
        ("# Model 3: LSTM"),
        ("## Model 3 Prediction vs. Actual"),
        model3_df.hvplot.line(y="Real", label="Act.")
        * model3_df.hvplot.line(y="Predicted", label="Pred."),
        ("## Model 3 Dataset"),
        model3_df.hvplot.table(
            title="Model 3 Predictions and Actual",
            columns=["Real", "Predicted"]
        )
    )

    dashboard = pn.Tabs(
        ("Introduction", tab_intro)
        ,("Linear Regression", tab_1_column)
        ,("ARIMA", tab_2_column)
        ,("LSTM", tab_3_column)
    )

    return dashboard


#Execute the Code
master_df = create_Master_DF()
correlation = master_df.corr()
stock_corr = correlation.loc[ : , ['VLO']].reset_index(drop=False, inplace=False)
stock_corr.rename(columns={'index': 'Features'}, inplace=True)
stock_corr['stock'] = 'VLO'
model_1_out_df = LRLB(master_df)
model_2_out_df = arima_fuc(master_df)
model_3_out_df = LSTM_F(master_df)
dashboard = build_dashboard(master_df, stock_corr, model_1_out_df, model_2_out_df, model_3_out_df)
dashboard.servable().show()
print("The End")
