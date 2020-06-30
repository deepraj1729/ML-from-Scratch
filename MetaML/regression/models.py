import pandas as pd 
import numpy as np
from sklearn import preprocessing,model_selection
import matplotlib.pyplot as plt 
from matplotlib import style 
import random
from statistics import mean
from matplotlib import style 



style.use('fivethirtyeight')

df = pd.read_csv('Sample dataset.csv')
print("__________________________________________")
print(df.head(10))
print("__________________________________________")
print(df.tail(10))

def plot_corr(df):
    corr = df.corr()
    fig,ax = plt.subplots(figsize = (6,6))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.show()
    
    
plot_corr(df) 

x = np.array(df.drop(['y_values'],1), dtype = float)                                         # input values or x-values
y = np.array(df['y_values'], dtype = float)                                                  # output values or y_values
x = x.flatten()

x_train ,x_test , y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.3)   #Splits data into 70:30 ratio

class Linear_Regression():                                                                   # Our own linear regression Model
    def __init__(self):
        print("The model is ready for use")
    
        
    def best_fit_slope_intercept(self,xs,ys):        # for finding best fit slope(m) and intercept(c)
        
        self.m = ( ( mean(xs) * mean(ys) -mean(xs*ys)  ) / ( pow(mean(xs),2) - mean(pow(xs,2)) ) )   # For slope (m)
        self.c = mean(ys) - self.m * mean(xs)                                                        # For y-intercept(c)
    
    def fit(self,xs,ys):
        
        self.best_fit_slope_intercept(xs,ys) 
        self.regression_line = [((self.m)*x)+ (self.c) for x in xs]
        print("Model fitting completed")
        return self.regression_line                                     # Returns the fitted values (ys_line) for each input (xs) 
    
    def squared_error(self,ys_orig, ys_line):                           # Calculates the mean squared error between y_orig and y_line
        return sum((ys_line-ys_orig)**2)                                # N.B.  ys_orig = ys , ys_line = self.regression_line


    def score(self,ys_orig,ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        sqr_err_reg = self.squared_error(ys_orig, ys_line)
        sqr_err_y_mean = self.squared_error(ys_orig, y_mean_line)
        return 1- (sqr_err_reg / sqr_err_y_mean )
    
    def predict(self,x_testing):
        s = self.m *(x_testing) + self.c
        return s 
print("__________________________________________")
model = Linear_Regression()
print("__________________________________________")
y_line = model.fit(x_train,y_train)
print("__________________________________________")
print("Model accuracy: {} % " .format(model.score(y_train,y_line)*100))
print("__________________________________________")
y_pred = np.round(model.predict(x_test))
print("y_pred: {} " .format(y_pred))
print("__________________________________________")
print("y_test: {}".format(y_test))
print("__________________________________________")


# Data Visualization:

a = plt.scatter(x_train,y_train,s=100, c ='blue')
plt.plot(x_train,y_line,c='red' )
b = plt.scatter(x_test,y_test,s=120,c ='green')
c = plt.scatter(x_test,y_pred,s =130, c ='magenta')
plt.legend((a,b,c),('Trained points','Tested points','Predicted points'))
plt.show()
