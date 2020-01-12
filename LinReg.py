# -*- coding: utf-8 -*-
"""

@author: Yskandar
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#matplotlib inline
import wget as wg

#Downloading the data file (csv)
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
FuelConsumption = wg.download(url)

#Showing the data
df= pd.read_csv("FuelConsumptionCo2.csv")
df.head


df.describe()
cdf = df[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


plt.figure(1)
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


plt.figure(2)
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red')
plt.xlabel("Cylinders")
plt.ylabel("Emissions")
plt.show()

#First let's build the train set

msk=np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk] # ~ changes all the boolean values to their opposite


#Showing the train set
plt.figure(3)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Sklearn package : model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y) #computing the regression coefficients

print('Coefficients:', regr.coef_)
print('Intercept: ', regr.intercept_)

#Plotting the fit line over the data to compare
plt.figure(4)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
 
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error :%.2f" %np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" %np.mean((test_y_-test_y)**2))
print("R2_score: %.2f" % r2_score(test_y_, test_y)) #scores the accuracy of the constructed model


