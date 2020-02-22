# -*- coding: utf-8 -*-
"""
@author: Yskandar Gas
"""

import numpy as np
import pandas as pd
import wget as wg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

#Downloading the data file (csv)
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv'
china_gdp = wg.download(url)


#Showing the data
df= pd.read_csv("china_gdp.csv")
df.head(10)

#Plotting the dataset
plt.figure()
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


"""
After plotting the Dataset, the logical function appears as a good approximation here

"""

#Model

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

"""
Indeed this model seems quite accurate, let's build the regression model
"""

#Build

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

xdata = x_data/max(x_data) #normalizing the data
ydata = y_data/max(y_data)

beta, pcov = curve_fit(sigmoid, xdata, ydata)
beta1 = beta[0]
beta2 = beta[1]
print(" beta_1 = {}, beta_2 = {}".format(beta1, beta2))


#Plotting the final regression model

x = np.linspace(1960, 2015, 55)
x = x/max(x) #normalizing
plt.figure()
y = sigmoid(x, beta1, beta2)

plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

"""
In order to evaluate our model, we have to spit the Dataset 
into a train set and a test set

"""


pos = np.random.rand(len(df))<0.8 #boolean values
train_x = xdata[pos]
train_y = ydata[pos]
test_x = xdata[~pos]
test_y = ydata[~pos]

betap, pcov = curve_fit(sigmoid, train_x, train_y)
betap1 = betap[0]
betap2 = betap[1]
model_y = sigmoid(test_x, betap1, betap2)

"""
Evaluation of the model
"""
MAE = np.mean(np.absolute(model_y - test_y))
print("MAE ={}".format(MAE))
MSE = np.mean((model_y - test_y)**2)
print("MSE = {}".format(MSE))
R2_score = r2_score(model_y, test_y)
print("R2 = {}".format(R2_score))



