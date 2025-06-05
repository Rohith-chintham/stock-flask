import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data(data, look_back=60):
    """
    Scales and prepares the dataset for LSTM training.

    Parameters:
        data (np.array): The stock price data.
        look_back (int): Number of previous time steps to use for prediction.

    Returns:
        x, y: Arrays formatted for LSTM input and output.
        scaler: The scaler object for inverse transforming later.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    x = np.array(x)
    y = np.array(y)

    return x, y, scaler

def build_model(input_shape):
    """
    Builds and compiles an LSTM model.

    Parameters:
        input_shape (tuple): Shape of input data (time steps, features).

    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
