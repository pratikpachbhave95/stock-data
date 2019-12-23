#####################
#IMPORTING LIBRARIES#
#####################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################
#IMPORTING THE TRAINING SET#
############################
dataset_train = pd.read_csv('D:\\stock\\Google\\google1_train.csv')
training_set = dataset_train.iloc[:, 1:2].values
dataset_train.head()

#################
#FEATURE SCALING#
#################
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled

X_train = [] 
y_train = []  
for i in range(1, len(training_set_scaled)): # upper bound is last row, lower bound is i-60
    X_train.append(training_set_scaled[i-1:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

############
# Reshaping#
############
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

##################
#BUILDING THE RNN#
##################

############################################
#IMPORTING THE KERAS LIBRARIES AND PACKAGES#
############################################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#######################
#ININTIALISING THE RNN#
#######################
regressor = Sequential()

#############################################################
#ADDING THE FIRST LSTM LAYER AND SOME DROPOUT REGULARISATION#
#############################################################
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

##############################################################
#ADDING THE SECOND LSTM LAYER AND SOME DROPOUT REGULARISATION#
##############################################################
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#############################################################
#ADDING THE THIRD LSTM LAYER AND SOME DROPOUT REGULARISATION#
#############################################################
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

##############################################################
#ADDING THE FOURTH LSTM LAYER AND SOME DROPOUT REGULARISATION#
##############################################################
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#########################
#ADDING THE OUTPUT LAYER#
#########################
regressor.add(Dense(units = 1))

###################
#COMPILING THE RNN#
###################
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#####################################
#FITTING THE RNN TO THE TRAINING SET#
#####################################
regressor.fit(X_train, y_train, epochs = 1000, batch_size = 32)

###################################
#GETTING REAL STOCK PRICES OF 2017#
###################################
dataset_test = pd.read_csv('D:\\stock\\Google\\google1_test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#############################################
#GETTING THE PREDICTED STOCK PRICES FOR 2017#
#############################################
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 1:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(1, 21):
    X_test.append(inputs[i-1:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

########################
#VISUALISING THE RESULT#
########################
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
