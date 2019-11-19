import pandas as pd
from datetime import datetime
from iexfinance.stocks import get_historical_data
import quandl

import bokeh
import numpy as np
import hvplot.pandas
import panel as pn

from LBLR import LRLB

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


def build_dashboard(model1_df):
    """Build the dashboard with 3 tabs - one of each model"""
    
    model2_df = model1_df
    model3_df = model1_df

    # Create tab_1_columns for Model 1
    tab_1_column = pn.Column(
        ("# Model 1: Linear Regression on Lagged Brent"),
        ("## Model 1 Prediction vs. Actual"),
        model1_df.hvplot.line(y="Actual Returns", label="Act. Rtn.")
        * model1_df.hvplot.line(y="Out-of-Sample Predictions", label="Pred. Rtn."),
        ("## Model 1 Dataset: Actual Returns vs. Predicted Returns."),
        model1_df.hvplot.table(
            title="Model 1: Predictions and Actual",
            columns=["Actual Returns", "Out-of-Sample Predictions"]
        )
    )

    # # Create tab_2_columns for Model 2
    # tab_2_column = pn.Column(
    #     ("# Model 2: xyz"),
    #     ("## Model 2 Prediction vs. Actual"),
    #     model2_df.hvplot.line(y="VLO", label="Act.")
    #     * model2_df.hvplot.line(y="S_P", label="Pred."),
    #     ("## Model 2 Dataset"),
    #     model2_df.hvplot.table(
    #         title="Model 2 Predictions and Actual",
    #         columns=["VLO", "S_P"]
    #     )
    # )

    # # Create tab_3_columns for Model 3
    # tab_3_column = pn.Column(
    #     ("# Model 3: xyz"),
    #     ("## Model 3 Prediction vs. Actual"),
    #     model3_df.hvplot.line(y="VLO", label="Act.")
    #     * model3_df.hvplot.line(y="S_P", label="Pred."),
    #     ("## Model 3 Dataset"),
    #     model3_df.hvplot.table(
    #         title="Model 3 Predictions and Actual",
    #         columns=["VLO", "S_P"]
    #     )
    # )

    dashboard = pn.Tabs(
        ("Linear Regression", tab_1_column)
        # ,("Model 2", tab_2_column)
        # ,("Model 3", tab_3_column)
    )

    return dashboard


#Execute the Code
test_df = create_Master_DF()
model1_df = LRLB(test_df)
dashboard = build_dashboard(model1_df)
dashboard.servable().show()
print("The End")
