import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def predict_sales(radio, weight, bias):
    return weight * radio + bias

def predict_sales_matrix(radio, weights, biases):
    return np.multiply(np.array(radio), weights) + biases


def cost_function(radio, sales, weight, bias):    
    companies = len(radio)
    total_error = ((np.array(sales) - np.array(predict_sales(radio, weight, bias))) ** 2).sum()
    return total_error / companies

def cost_function_matrix(radio, sales, weights, biases):
    total_errors = []
    for i in range(0, len(weights)):
        for j in range(0, len(biases)):
            total_errors.append(cost_function(radio, sales, weights[i], biases[j]))

    return np.array(total_errors)


def update_weights(radio, sales, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(radio)

    for i in range(companies):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*radio[i] * (sales[i] - (weight*radio[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(sales[i] - (weight*radio[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / companies) * learning_rate
    bias -= (bias_deriv / companies) * learning_rate

    return weight, bias

def train(radio, sales, weight, bias, learning_rate, iters):
    cost_history = []
    weight_history = []
    bias_history = []

    for i in range(iters):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append(cost)
        weight_history.append(weight)
        bias_history.append(bias)

        # Log Progress
        if i % 5 == 0:
            print("iter: "+str(i) + " cost: "+str(cost))
            plt.scatter(radio, sales)
            plt.plot(radio, [y * weight + bias for y in radio], color='red')
            plt.title('Iter: %i    Weight: %f\nBias: %f    Cost: %f\n' % (i, weight, bias, cost))
            plt.show()

        
    return weight, bias, cost_history, weight_history, bias_history

def cost_function_plot3D(radio, sales):
    weights = np.linspace(-2, 2, 1000)
    biases = np.linspace(-2, 2, 1000)

    weights, biases = np.meshgrid(weights, biases)

    costs = cost_function_matrix(radio, sales, weights, biases)

    ax = plt.axes(projection='3d')
    ax.plot3D(weights.ravel(), biases.ravel(), costs.ravel())
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_zlabel('Cost')
    plt.show()

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
    




if __name__ == '__main__':
    
    df = pd.read_csv('Advertising.csv')

    # Get sales and radio data from csv
    y = df['sales']
    x = df['radio']

    n = 100

    w = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)


    # print(predict_sales(x, w[0], b[0]))
    # print(predict_sales(x, w[0], b[1]))
    # print(predict_sales(x, w[0], b[2]))

    weights, biases = np.meshgrid(w, b)
    weights = weights.ravel()
    biases = biases.ravel()
    print(weights)
    print(biases)
    costs = []
    for i in range(0, n ** 2):
            cost = cost_function(x, y, weights[i], biases[i])
            costs.append(cost)
    
    print(len(costs))

    costs = np.array(costs)
    print(costs)

    Z = f(w,b)
    print(Z.shape)

    print(weights.shape)
    print(biases.shape)
    print(costs.shape)

    weights = weights.reshape(n, n)
    biases = biases.reshape(n, n)
    costs = costs.reshape(n, n)

    ax = plt.axes(projection='3d')
    ax.plot_wireframe(weights, biases, costs, color='grey')
    ax.scatter(weights[10][10], biases[10][10], costs[10][10], color='red')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_zlabel('Cost')
    plt.show()

    

    # Plot data
    plt.scatter(x, y)
    plt.xlabel('Radio ($)')
    plt.ylabel('Sales (unit)')
    plt.show()

    weight = 0.3
    bias = 0.014

    # Chose random weight and bias for initial step
    weight = np.random.randn()
    bias = np.random.randn()

    # Print random weight and bias in a console
    print('Random weight:', weight)
    print('Random bias:', bias)

    learning_rate = 0.001
    iters = 100

    weight, bias, cost_history, weight_history, bias_history = train(x, y, weight, bias, 0.001, 100)
    # print(weight, bias, cost_history)

    cost_function_plot3D(x, y)

    # plt.plot(weight_history, bias_history, cost_history)
    # plt.show()

    plt.plot(range(0, iters), cost_history)
    plt.title('Cost Function')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.show()


# for i in range(0, 30):
#     print('Iter:', i, 'Weight:', weight, 'Bias:', bias)
#     print('Cost:', cost_function(x, y, weight, bias))
#     weight,bias = update_weights(x, y, weight, bias, 0.001)

    
# plt.xlim(xmin=-5, xmax=60)
# plt.ylim(ymin=-5, ymax=50)
# plt.plot(x, predict_sales(x, 0.45, 0.025))
    

# print(cost_function(x, y, 0.45, 0.025))

   

# plt.scatter(x, y)
# plt.plot(x, x * weight + bias)

# def linear_function(pats, w, b):
#     return [(x * w) + b for x in pats]

# def error_function(wags, wags_fit):
#     return ((np.array(wags) - np.array(wags_fit)) ** 2).sum() / len(wags)

 #pats = [1, 2, 4]
 #wags = [2, 4, 5]


# w = np.random.randn()
# b = np.random.randn()

# print('w:', w)
# print('b:', b)

# w_list = np.arange(-10, 10, 0.1)
# b_list = np.arange(-10, 10, 0.1)

# yfit = linear_function(pats, w, b)

# print('MSE:', error_function(wags, yfit))

# plt.scatter(pats, wags)
# plt.plot(pats, yfit)
# plt.show()