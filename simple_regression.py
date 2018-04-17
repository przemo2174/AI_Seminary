import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def predict_sales(radio, weight, bias):
    return weight * radio + bias

def cost_function(radio, sales, weight, bias):    
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / companies

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

    for i in range(iters):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iter: "+str(i) + " cost: "+str(cost))

        # plt.scatter(radio, sales)
        # plt.plot(radio, [y * weight + bias for y in radio])
        # plt.show()

    return weight, bias, cost_history

df = pd.read_csv('Advertising.csv')

y = df['sales']
x = df['radio']

plt.scatter(x, y)
plt.xlabel('Radio ($)')
plt.ylabel('Sales (unit)')
# plt.xlim(xmin=-5, xmax=60)
# plt.ylim(ymin=-5, ymax=50)
plt.plot(x, predict_sales(x, np.random.randn(), np.random.randn()))
plt.show()

# weight, bias, cost_history = train(x, y, np.random.randn(), np.random.randn(), 0.1, 500)

# plt.scatter(x, y)
# plt.plot(x, x * weight + bias)

# def linear_function(pats, w, b):
#     return [(x * w) + b for x in pats]

# def error_function(wags, wags_fit):
#     return ((np.array(wags) - np.array(wags_fit)) ** 2).sum() / len(wags)

# pats = [1, 2, 4]
# wags = [2, 4, 5]

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