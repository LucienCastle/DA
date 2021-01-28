# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
# Reproduceable Dataset
X, y = make_regression(n_samples=8, n_features=1, n_informative=1, noise=150, bias=50, random_state=200)
m = 8

# X = pd.DataFrame(np.array([10,9,2,15,10,16,11,16]))
# y = pd.DataFrame(np.array([95,80,10,50,45,98,38,93]))
# m = 8

def hypothesis(X,w):
    return (w[1]*np.array(X[:,0])+w[0])
def cost(w,X,y):
    return (.5/m) * np.sum(np.square(hypothesis(X,w)-np.array(y)))
def grad(w,X,y):
    g = [0]*2
    g[0] = (1/m) * np.sum(hypothesis(X,w)-np.array(y))
    g[1] = (1/m) * np.sum((hypothesis(X,w)-np.array(y))*np.array(X[:,0]))
    return g
def descent(w_new, w_prev, lr):
    # print(w_prev)
    # print(cost(w_prev,X,y))
    i=0
    while True:
        w_prev = w_new
        w0 = w_prev[0] - lr*grad(w_prev,X,y)[0]
        w1 = w_prev[1] - lr*grad(w_prev,X,y)[1]
        w_new = [w0, w1]
        # print(w_new)
        # print(cost(w_new,X,y))
        if (w_new[0]-w_prev[0])**2 + (w_new[1]-w_prev[1])**2 <= pow(10,-6):
            return w_new
        if i>700: 
            return w_new
        i+=1
        draw(X,w0+w1*X)
        
def draw(x1,x2):
    # _,ax1=plt.subplots(figsize=(10,10))
    # ax1.scatter(X,y,color='r')
    ln=plt.plot(x1,x2)
    plt.pause(0.001)
    ln[0].remove()
        
# def graph(formula, x_range):  
#     x = np.array(x_range)  
#     y = my_formula(x)  
#     plt.plot(x, y)  
    
def my_formula(x):
    return w[0]+w[1]*x
def r2_test(y,y_calc):
    SSR = np.sum((y-y_calc)**2)
    SST = np.sum((y-y.mean())**2)
    return (1-(SSR/SST))*100
# Training Model
w = [0,0]
_,ax1=plt.subplots(figsize=(10,10))
ax1.scatter(X,y,color='r')
w = descent(w,w,.1)
# Plotting Results
# plt.scatter(X,y,color = 'red', alpha = 0.2)
# graph(my_formula, range(-4,5))
# plt.xlabel('X')
# plt.ylabel('Y')
plt.show()
# R² Test
print(r2_test(y,hypothesis(X,w)))
