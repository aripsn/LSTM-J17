# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:17:50 2023

@author: Nautilus
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

dir = 'F:/Semester 1 Spring 2023/Machine Learning for CVEN/Project_5'
os.chdir(dir)

# import dataset
df = pd.read_csv('j17.csv') # if it's csv file
#df = pd.read_excel('F:/Semester 1 Spring 2023/Water Quality Modeling/Water_Q_Flow_15m.xlsx') # if it's excel file
df=df.drop(['Site'], axis=1)

# convert datetime column to datetime type 'datetime64[ns]'
df['DailyHighDate'] = pd.to_datetime(df['DailyHighDate'].astype(str)) # 'Date' is the column name
# API reference: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
df.head(10)

#df.isnull().values.any()
# set datetime as index
df.index = df['DailyHighDate']
df=df.drop(['DailyHighDate'], axis=1)
df=df.interpolate()
# resort by datetime index
df.sort_index(ascending=True)
df.head(10)

plt.figure(figsize=(18,6))
plt.plot(df['WaterLevelElevation'])
plt.legend(['J17 data'])

scaler = MinMaxScaler()
# fit the format of the scaler
data = df['WaterLevelElevation'].values.reshape(-1, 1)
scaled_df = scaler.fit_transform(data)

seq_len = 30
window = seq_len-1

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(scaled_df, seq_len, train_frac):
    sequences = split_into_sequences(scaled_df, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    n_val = int(sequences.shape[0] * 0.3)
    x_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    
    x_val = sequences[n_train:(n_train+n_val), :-1, :]
    y_val = sequences[n_train:(n_train+n_val), -1, :]
    
    x_test = sequences[(n_val+n_train):, :-1, :]
    y_test = sequences[(n_val+n_train):, -1, :]
    return x_train, y_train, x_test, y_test, x_val, y_val

x_train, y_train, x_test, y_test, x_val, y_val = get_train_test_sets(scaled_df, seq_len, train_frac=0.5)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(window, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head(5))
print(hist.tail(5))

plt.figure(figsize=(6,4))

plt.plot(hist['epoch'], hist['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()        


# Make predictions on the test data
y_pred = model.predict(x_test)
y_trainP = model.predict(x_train)
# Inverse scale the predictions and actual values
y_pred = scaler.inverse_transform(y_pred)
y_trainP = scaler.inverse_transform(y_trainP)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.reshape(-1,1))

# Calculate the root mean squared error (RMSE) of the predictions
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f'RMSE: {rmse:.2f}')

#plt.figure(figsize=(18,6))
plt.plot(y_test, color = 'red',  label = 'Actual Data') #linewidth=1, markersize=6,
plt.plot(y_pred, color = 'green',  label = 'Predicted Data') #linewidth=1, markersize=6,
plt.legend(loc='best')
plt.show()

#plt.figure(figsize=(18,6))
plt.plot(y_train, label = 'Actual Data')
plt.plot(y_trainP, linewidth=1, label = 'Predicted Data')
plt.legend(loc='best')
plt.show()
