# Import libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:\\1\\00\\Salary.txt'
data = pd.read_csv(path, header = None, names = ['Years_Experience', 'Salary'])

print(data.head(10))
print('Data Description =', data.describe())
print('**************************')

data.plot(kind = 'scatter', x = 'Years_Experience', y = 'Salary', figsize = (5,5))

data.insert(0, 'Ones', 1) # 0 = first column, ones =  name of column, 1 = column filled with 1s as values
print('new data = \n', data.head(10))
print('**************************')

cols = data.shape[1] # shape of data is (97 x 3), then shape[0] = 97 and shape[1] = 3
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
print('**************************************')
print('X data = \n' ,X.head(10) )
print('y data = \n' ,y.head(10) )
print('**************************************')

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print('X \n',X)
print('X.shape = ' , X.shape)
print('theta \n',theta)
print('theta.shape = ' , theta.shape)
print('y \n',y)
print('y.shape = ' , y.shape)
print('**************************************')

# cost function
def computeCoste(X, y, theta):
    z = np.power(((X*theta.T) - y), 2)
#    print('z \n',z)
#    print('m ' ,len(X))
    return np.sum(z) / (2 * len(X))

# print('computeCost(X, y, theta) = ' , computeCoste(X, y, theta))

print('**************************************')

# GD function

def GradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape)) # [0  0]
    parameters = int(theta.ravel().shape[1]) #theta.shape[1]=2(theta.shape = 1x2), parameters = 2
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X*theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCoste(X, y, theta)
        
    return theta, cost


# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000 # you can add the number of iterations to make sure that you get the minimized cost

# perform gradient descent to "fit" the model parameters
g, cost = GradientDescent(X, y, theta, alpha, iters)

print('g = ' , g) # values of theta0 and theta1
print('cost  = ' , cost[0:50] )
print('computeCost = ' , computeCoste(X, y, g))
print('**************************************')

###############################################################################



# get best fit line

x = np.linspace(data.Years_Experience.min(), data.Years_Experience.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x) # same as h(x) = theta0 + theta1 * x
print('f \n',f)


    
# draw the line

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Years_Experience, data.Salary, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Years_Experience')
ax.set_ylabel('Salary')
ax.set_title('Predicted Salary vs. Years_Experience Size')





# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')



