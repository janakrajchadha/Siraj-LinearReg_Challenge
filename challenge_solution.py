# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:23:29 2017

@author: Janak
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

def rmse(x_values,y_values,fit_line):
    pred=[fit_line.predict(x.reshape(-1,1)) for x in x_values.values]
    squares=[(y-prediction)**2 for y,prediction in zip(y_values.values,pred)]
    
    rmse=np.sqrt(sum(squares)/(len(x_values)-2))
    return rmse    

#read data
dataframe = pd.read_fwf('challenge_dataset.txt',header=None)
dataframe.columns=['x','y']
x_values = dataframe[['x']]
y_values = dataframe[['y']]

#train model on data
fit_line= linear_model.LinearRegression()
fit_line.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, fit_line.predict(x_values))
plt.show()

err=rmse(x_values,y_values,fit_line)

print("The best fit line is : y = {0}x + {1}".format(fit_line.coef_[0][0], fit_line.intercept_[0]))
print("The RMSE calculated is {0}".format(err))
