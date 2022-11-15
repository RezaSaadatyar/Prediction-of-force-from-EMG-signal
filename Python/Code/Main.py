# ==========================================================================
# ===================EMG force & Torque estimation  ========================
# ====================== Presented by: Reza Saadatyar  =====================
# =================== E-mail: Reza.Saadatyar92@gmail.com  ==================
# ============================  2022-2023 ==================================
# The program will run automatically when you run code/file Main.py, and you do not need to run any of the other codes.
# ============================================= Import Libraries ========================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
from AR_Model import ar
from ARX_Model import arx
from scipy import io, signal
from AR_LS_Model import ar_ls
import matplotlib.pyplot as plt
from Filtering import filtering
from keras import models, layers
from Plot_Models import plot_models
from Least_Squares import lest_squares
from Sequences_Data import sequences_data
from Linear_Regression import linear_regression
from Xgboost_Regression import xgboost_regression
from Random_Forest_Regression import random_forest_regression
from Tree_Decision_Regression import tree_decision_regression
# ============================================Step 1: Preparing the data ===========================================
os.system('cls')
Data = io.loadmat('50.mat')
print(Data.keys())
Raw_Torque = Data['Raw_Torque'].flatten()
TAB_RMS = Data['TAB_RMS']
BB = np.array(TAB_RMS['BB'].tolist()).flatten()   # Biceps Brachii
BR = np.array(TAB_RMS['BR'].tolist()).flatten()   # Brachioradialis
TBM = np.array(TAB_RMS['TM'].tolist()).flatten()  # Triceps Brachii medial
TBL = np.array(TAB_RMS['TL'].tolist()).flatten()  # Triceps Brachii lateral
Input_Data = pd.Series(BB+BR+TBM+TBL)
# ======================================== Step 2: Filtering & Downsampling ======================================
Filter_Torque = filtering(Raw_Torque, F_low=1, F_high=0, Order=2, Fs=2048, btype='low')      # btype:'low', 'high', 'bandpass', 'bandstop'
Filter_Torque = pd.Series(signal.decimate(Filter_Torque, 1024, ftype="fir"))
if len(Filter_Torque) > len(Input_Data):
    Filter_Torque = Filter_Torque[0:len(Filter_Torque)-1]
# =========================== Step 3: Split Dataset intro Train and Test =========================================
nLags = 3
num_sample = 200
mu = 0.000001
Data_Lags = pd.DataFrame(np.zeros((len(Input_Data), nLags)))

for i in range(0, nLags):
    Data_Lags[i] = Input_Data.shift(i + 1)
Data_Lags = Data_Lags[nLags:]
data = Filter_Torque[nLags:]
Data_Lags.index = np.arange(0, len(Data_Lags), 1, dtype=int)
data.index = np.arange(0, len(data), 1, dtype=int)
train_size = int(len(data) * 0.8)
# ============================================= Plot ==========================================================
sns.set(style='white')
fig, axs = plt.subplots(nrows=4, ncols=1, sharey='row', figsize=(16, 10))
plot_models(pd.DataFrame({'Torque': Filter_Torque, 'BB': BB, 'BR': BR, 'TBM': TBM, 'TBL': TBL}), [], [], axs, nLags, train_size, num_sample=num_sample, type_model='Actual_Data')
# ================================= Step 5: Autoregressive and Automated Methods ===============================
# -------------------------------------------  Least Squares ---------------------------------------------------
lest_squares(data, Data_Lags, train_size, axs, num_sample=num_sample)
# -------------------------------------------- Auto-Regressive (AR) model --------------------------------------
ar(data, Data_Lags, train_size, axs, mu=mu, num_sample=num_sample)
# ----------------------------------------- Auto-Regressive (AR) + LS model ------------------------------------
ar_ls(data, Data_Lags, train_size, axs, mu=mu, num_sample=num_sample)
# ------------------------------------------------  ARX --------------------------------------------------------
arx(data, Data_Lags, train_size, axs, mu=mu, num_sample=num_sample)
# ======================================= Step 5: Machine Learning Models ======================================
# ------------------------------------------- Linear Regression Model  -----------------------------------------
linear_regression(data, Data_Lags, train_size, axs, num_sample=num_sample)
# ------------------------------------------ RandomForestRegressor Model ---------------------------------------
random_forest_regression(data, Data_Lags, train_size, axs, n_estimators=1000, max_features=nLags, num_sample=num_sample)
# -------------------------------------------- Decision Tree Model ---------------------------------------------
tree_decision_regression(data, Data_Lags, train_size, axs, max_depth=3, num_sample=num_sample)
# ---------------------------------------------- xgboost -------------------------------------------------------
xgboost_regression(data, Data_Lags, train_size, axs, n_estimators=1000, num_sample=num_sample)
# -----------------------------------------------  LSTM model --------------------------------------------------
train_x, train_y = sequences_data(np.array(data[:train_size]), nLags)  # Convert to a time series dimension:[samples, nLags, n_features]
test_x, test_y = sequences_data(np.array(data[train_size:]), nLags)
mod = models.Sequential()  # Build the model
# mod.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', input_shape=(None, nLags)))  # ConvLSTM2D
# mod.add(layers.Flatten())
mod.add(layers.LSTM(units=100, activation='tanh', input_shape=(None, nLags)))
mod.add(layers.Dropout(rate=0.3))
# mod.add(layers.LSTM(units=100, activation='tanh'))  # Stacked LSTM
# mod.add(layers.Bidirectional(layers.LSTM(units=100, activation='tanh'), input_shape=(None, 1)))     # Bidirectional LSTM: forward and backward
mod.add(layers.Dense(32))
mod.add(layers.Dense(1))   # A Dense layer of 1 node is added in order to predict the label(Prediction of the next value)
mod.compile(optimizer='adam', loss='mse')
mod.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=2, epochs=100)
y_train_pred = pd.Series(mod.predict(train_x).ravel())
y_test_pred = pd.Series(mod.predict(test_x).ravel())
y_train_pred.index = np.arange(nLags, len(y_train_pred)+nLags, 1, dtype=int)
y_test_pred.index = np.arange(train_size + nLags, len(data), 1, dtype=int)
plot_models(data, y_train_pred, y_test_pred, axs, nLags, train_size, num_sample=num_sample, type_model='LSTM')
# data_train = normalize.inverse_transform((np.array(data_train)).reshape(-1, 1))
mod.summary(), plt.tight_layout(), plt.subplots_adjust(wspace=0, hspace=0.2), plt.show()
