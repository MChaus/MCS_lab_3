
import numpy as np
from numpy import linalg as LA

class Lasso(object):
    def __init__(self, X, y, lambda_val=1, eps=0.001, intercept=True, normalize=True):
        self.X = np.ones((X.shape[0], X.shape[1] + 1))
        if normalize:
            self.X[:,1:] = np.copy((X - self.x_mean) / self.x_std)
        else:
            self.X[:,1:] = np.copy(X)
        if not intercept:
            self.X = self.X[:,1:]
        self.y = np.copy(y)
        self.n, self.m = self.X.shape
        self.theta = np.ones((self.m, 1))
        self.lambda_val = lambda_val
        self.eps = eps

    def cost_functions_gradient(self):
        grad = np.zeros((self.m, 1))
        for j, theta in enumerate(self.theta):
            grad[j] = -1 / self.n * sum([(y - x.dot(self.theta)) * x[j] for i, (x, y) in enumerate(zip(self.X, self.y))]) +\
            self.lambda_val * np.sign(theta)
        return grad

    def cost_function(self, theta=None):
        if theta is None:
            theta = self.theta
        return 1 / (self.n * 2) * np.dot((self.X.dot(theta) - self.y).T, (self.X.dot(theta) - self.y)) +\
        self.lambda_val * sum(np.absolute(theta))

    def gradien_descent(self, alpha=1000, arg_condition=True,
                        grad_condition=True, cost_func_condition=True):
        delta_theta = 1
        delta_cost_function = 1
        prev_cost_function = self.cost_function(self.theta)
        while ((LA.norm(delta_theta) > self.eps and arg_condition == True) or
               (delta_cost_function > self.eps and cost_func_condition == True) or
               (LA.norm(self.cost_functions_gradient()) > self.eps and grad_condition == True)):
            delta_theta = alpha * self.cost_functions_gradient()
            next_cost_function = self.cost_function(self.theta -  delta_theta)
            while (prev_cost_function < next_cost_function):
                delta_theta /= 2
                next_cost_function = self.cost_function(self.theta -  delta_theta)
            prev_cost_function = next_cost_function
            delta_cost_function = prev_cost_function - next_cost_function
            self.theta -=  delta_theta
        return self

    def MSE(self):
        return 1 / self.n * np.dot((self.X.dot(self.theta) - self.y).T, (self.X.dot(self.theta) - self.y))

if __name__ == '__main__':
     X = np.array([[3, 1, 2], [4, -3, -2.5], [10, 0, -4], [-5, 1.2, 3]])
     y = np.array([10.3, -9.21, -1.9, 6.48]).reshape(-1, 1)
     obj1 = Lasso(X, y, lambda_val=0, eps=0.0001, normalize=False, intercept=False)
     obj1.gradien_descent(arg_condition=True,
                         grad_condition=True, cost_func_condition=True)
     print(LA.inv(obj1.X.T @ obj1.X) @ obj1.X.T @ obj1.y)
     print(obj1.theta)
     print(obj1.MSE())
