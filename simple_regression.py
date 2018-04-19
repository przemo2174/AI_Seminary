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
    global weights
    global biases
    global costs
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

            # plot data
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1.scatter(radio, sales)
            ax1.plot(radio, [y * weight + bias for y in radio], color='red')
            ax1.set_title('Iter: %i    Weight: %f\nBias: %f    Cost: %f\n' % (i, weight, bias, cost))
            ax2 = fig.add_subplot(212, projection='3d')
            ax2.plot_wireframe(weights, biases, costs, color='grey')
            ax2.scatter(weight, bias, cost, color='red')
            ax2.text(weight, bias, cost, 'Cost: %f\nWeight: %f\n Bias: %f' % (cost, weight, bias), size=12, color='green')
            ax2.set_xlabel('Weight')
            ax2.set_ylabel('Bias')
            ax2.set_zlabel('Cost')
          
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()

            # plt.scatter(radio, sales)
            # plt.plot(radio, [y * weight + bias for y in radio], color='red')
            # plt.title('Iter: %i    Weight: %f\nBias: %f    Cost: %f\n' % (i, weight, bias, cost))
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()
            # plt.show()

        
    return weight, bias, cost_history, weight_history, bias_history

def cost_function_plot3D(x, y, n):
    w = np.linspace(-5, 5, n)
    b = np.linspace(-30, 30, n)

    # Generate meshgrid
    weights, biases = np.meshgrid(w, b)
    weights = weights.ravel()
    biases = biases.ravel()

    costs = []
    for i in range(0, n ** 2):
            cost = cost_function(x, y, weights[i], biases[i])
            costs.append(cost)

    weights = weights.reshape(n, n)
    biases = biases.reshape(n, n)
    costs = np.array(costs).reshape(n, n)

    ax = plt.axes(projection='3d')
    ax.plot_wireframe(weights, biases, costs, color='grey')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_zlabel('Cost')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    
if __name__ == '__main__':
    
    # Read data from csv
    df = pd.read_csv('Advertising.csv')

    # Retrieve sales and radio data
    y = df['sales']
    x = df['radio']

    # Plot data x = radio, y = sales
    plt.scatter(x, y)
    plt.xlabel('Radio ($)')
    plt.ylabel('Sales (unit)')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

    # Chose random weight and bias for initial step
    weight = np.random.randn()
    bias = np.random.randn()

    # Print random weight and bias in a console
    print('Random weight:', weight)
    print('Random bias:', bias)

    # Initialize neccessary parameters
    learning_rate = 0.001
    iters = 100

    cost_function_plot3D(x, y, 100)

    # Generate weight and bias values needed for visualizing a cost function
    n = 100
    w = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)

    # Generate meshgrid
    weights, biases = np.meshgrid(w, b)
    weights = weights.ravel()
    biases = biases.ravel()

    costs = []
    for i in range(0, n ** 2):
            cost = cost_function(x, y, weights[i], biases[i])
            costs.append(cost)

    weights = weights.reshape(n, n)
    biases = biases.reshape(n, n)
    costs = np.array(costs).reshape(n, n)


    # Perform training
    weight, bias, cost_history, weight_history, bias_history = train(x, y, weight, bias, learning_rate, iters)

    # Plot cost in each iteration
    plt.plot(range(0, iters), cost_history)
    plt.title('Cost in each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()