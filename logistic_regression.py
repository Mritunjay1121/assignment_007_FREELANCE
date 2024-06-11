import numpy as np
import argparse
from sklearn.metrics import accuracy_score

# def sigmoid(z):
#     return '''complete code here'''

# def logistic_regression_newton(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6):
#     n_samples, n_features = X.shape
#     weights = np.zeros(n_features)
#     bias = 0

#     for _ in range(n_iterations):
#         linear_model = '''complete code here'''
#         y_predicted = '''complete code here'''

#         # Compute gradient
#         gradient = '''complete code here'''

#         # Compute Hessian matrix
#         hessian = '''complete code here'''

#         # Update parameters using Newton's method
#         '''complete code here'''


#     linear_model = np.dot(X, weights) + bias
#     y_predicted = sigmoid(linear_model)
#     y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
#     return np.array(y_predicted_cls)

# def logistic_regression_gradient_ascent(X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6):
#     n_samples, n_features = X.shape
#     weights = np.zeros(n_features)
#     bias = 0

#     for _ in range(n_iterations):
#         linear_model = '''complete code here'''
#         y_predicted = '''complete code here'''

#         # Compute gradient
#         gradient = '''complete code here'''

#         # Update parameters using gradient ascent
#         '''complete code here'''


#     linear_model = np.dot(X_test, weights) + bias
#     y_predicted = sigmoid(linear_model)
#     y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
#     return np.array(y_predicted_cls)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Logistic Regression with Newton's Method or Gradient Ascent")
#     parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
#     parser.add_argument("--test_data", type=str, help="Path to data file (CSV format)")
#     parser.add_argument("--method", type=str, default="newton", choices=["newton", "gradient"], help="Optimization method (newton or gradient)")
#     args = parser.parse_args()

#     if args.data:
#         data = np.genfromtxt(args.data, delimiter=',')
#         test_data = np.genfromtxt(args.test_data, delimiter=',')

#         X = data[1:, :-1]
#         y = data[1:, -1]
#         X_test = test_data[1:, :-1]
#         y_test = test_data[1:, -1]
#         if args.method == "newton":
#             predictions = logistic_regression_newton(X_test, y_test)
#         else:
#             predictions = logistic_regression_gradient_ascent(X_test, y_test)

#         print("Predictions:", predictions)
#     else:
#         print("Please provide the path to the training data file using the '--data' argument and testing data file using the '--test_data' argument.")


import numpy as np
import argparse

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_newton(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6):


    """
    Perform logistic regression using Newton's method.

    Args:
    - X (ndarray): Feature matrix.
    - y (ndarray): Target values.
    - learning_rate (float): Step size for updating weights.
    - n_iterations (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.

    Returns:
    - ndarray: Predicted target values.
    """




    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Compute gradient
        gradient = np.dot(X.T, (y_predicted - y)) / n_samples

        # Compute Hessian matrix
        hessian = np.dot(X.T, np.dot(np.diag(y_predicted * (1 - y_predicted)), X)) / n_samples

        # Update parameters using Newton's method
        weights -= learning_rate * np.linalg.inv(hessian).dot(gradient)

    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

def logistic_regression_gradient_ascent(X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6):
    
    
    """
    Perform logistic regression using gradient ascent.

    Args:
    - X (ndarray): Feature matrix.
    - y (ndarray): Target values.
    - learning_rate (float): Step size for updating weights.
    - n_iterations (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.

    Returns:
    - ndarray: Predicted target values.
    """

    
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Compute gradient
        gradient = np.dot(X.T, (y_predicted - y)) / n_samples

        # Update parameters using gradient ascent
        weights += learning_rate * gradient

    linear_model = np.dot(X_test, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression with Newton's Method or Gradient Ascent")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--test_data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--method", type=str, default="newton", choices=["newton", "gradient"], help="Optimization method (newton or gradient)")
    args = parser.parse_args()

    if args.data and args.test_data:
        data = np.genfromtxt(args.data, delimiter=',')
        test_data = np.genfromtxt(args.test_data, delimiter=',')

        X = data[1:, :-1]
        y = data[1:, -1]
        X_test = test_data[1:, :-1]
        y_test = test_data[1:, -1]
        
        if args.method == "newton":
            predictions = logistic_regression_newton(X_test, y_test)
        else:
            predictions = logistic_regression_gradient_ascent(X_test, y_test)

        print("Predictions:", predictions)
        print()
        print("Accuracy is :",accuracy_score(list(y_test),list(predictions)))
    else:
        print("Please provide the path to the training data file using the '--data' argument and testing data file using the '--test_data' argument.")
