import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

st.title("Stock Prediction Over Time")
st.sidebar.header("Select Stock")

tickers=['SPY', 'GLD', 'AAPL', 'VTI', 'QQQ']
selected_ticker = st.sidebar.selectbox("Choose a stock", tickers)

start=dt.datetime(2010,1,1)
end=dt.datetime.today()

data=yf.download(selected_ticker, start=start, end=end)
st.write(f"Showing historical data for {selected_ticker}")
st.write(data.tail())

st.subheader(f"{selected_ticker} Closing Price Over Time")
fig,ax=plt.subplots()
ax.plot(data.index, data["Close"], label=selected_ticker, color ='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
st.pyplot(fig)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Close'].values.reshape(-1,1))

train_size=int(len(scaled_data)*0.8)
train_data=scaled_data[:train_size]
test_data=scaled_data[train_size:]

def create_seqences(dataset, sequence_length=60):
    X,y=[],[]
    for i in range (len(dataset)-sequence_length):
        X.append(dataset[i:i+sequence_length])
        y.append(dataset[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length=60
X_train, y_train = create_seqences(train_data, sequence_length)
X_test, y_test = create_seqences(test_data, sequence_length)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

model=Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

st.subheader("Training Model....")
model.fit(X_train, y_train, epochs=10, batch_size=32)

predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)

st.subheader(f"{selected_ticker} Stock Price Prediction")
fig,ax=plt.subplots()
ax.plot(data.index[train_size+sequence_length:], data["Close"].values[train_size+sequence_length:], label="Actual Price", color="blue")
ax.plot(data.index[train_size+sequence_length:], predictions, label="Predicted Price", color ='red')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)