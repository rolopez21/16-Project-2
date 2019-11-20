from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout    

def window_data(df, window, feature_col_number, target_col_number):
        X = []
        y = []
        for i in range(len(df) - window - 1):
            features = df.iloc[i:(i + window), feature_col_number]
            target = df.iloc[(i + window), target_col_number]
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)


def LSTM_F(df):
    #Set window size
    window_size = 3
    # Column index 1 is the `Close` column
    feature_column = 15
    target_column = 15
    X, y = window_data(df, window_size, feature_column, target_column)
    
    # Use 70% of the data for training and the remainder for testing
    split = int(0.7 * len(X))
    X_train = X[: split - 1]
    X_test = X[split:]
    y_train = y[: split - 1]
    y_test = y[split:]
    
    # Use MinMaxScaler to scale the data between 0 and 1. 
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(y)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential()

    number_units = 10
    dropout_fraction = 0.2

    # Layer 1
    model.add(LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1))
        )
    model.add(Dropout(dropout_fraction))
    # Layer 2
    model.add(LSTM(units=number_units, return_sequences=True))
    model.add(Dropout(dropout_fraction))
    # Layer 3
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))
    # Output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    #train the model
    model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=1, verbose=1)    
   
    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))    
    
    #model_eval = model.evaluate(X_test, y_test)
       
    stocks = pd.DataFrame({"Real": real_prices.ravel(),"Predicted": predicted_prices.ravel()})
       
    return stocks  
