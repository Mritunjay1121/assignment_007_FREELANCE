import numpy as np
import argparse
from sklearn.metrics import r2_score

# class LinearRegressionGradientDescent:
#     def __init__(self, learning_rate=0.01, n_iterations=1000):
#         self.learning_rate = learning_rate
#         self.n_iterations = n_iterations
#         self.weights = None
#         self.bias = None

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0

#         for _ in range(self.n_iterations):
#             linear_model = '''complete code here'''
#             cost = '''complete code here'''

#             # Compute gradients
#             '''complete code here'''

#             # Update parameters
#             '''complete code here'''



#     def predict(self, X):
#         return np.dot(X, self.weights) + self.bias


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Linear Regression with Gradient Descent")
#     parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
#     parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for gradient descent")
#     parser.add_argument("--n_iterations", type=int, default=1000, help="Number of iterations for gradient descent")
#     args = parser.parse_args()

#     if args.data:
#         # Load data
#         data = np.genfromtxt(args.data, delimiter=',')
#         X = data[1:24, :-1]
#         y = data[1:24, -1]
#         X_test = data[25:, :-1]
#         # Initialize and train the model
#         model = LinearRegressionGradientDescent(learning_rate=args.learning_rate,
#                                                  n_iterations=args.n_iterations,
#                                                 )
#         model.fit(X, y)


#         # Make predictions
#         predictions = model.predict(X_test)

#         print("Predictions:", predictions)
#     else:
#         print("Please provide the path to the data file using the '--data' argument.")




import numpy as np
import argparse

class LinearRegressionGradientDescent:




    """
        Initialize Linear Regression model with Gradient Descent.

        Args:
        - learning_rate (float): Step size for gradient descent.
        - n_iterations (int): Number of iterations for gradient descent.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None



    """
        Train the linear regression model using gradient descent.

        Args:
        - X (ndarray): Feature matrix.
        - y (ndarray): Target values.
    """


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            cost = np.mean((linear_model - y) ** 2)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (linear_model - y))
            db = (1 / n_samples) * np.sum(linear_model - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):


        """
        Predict target values using the trained model.

        Args:
        - X (ndarray): Feature matrix.

        Returns:
        - ndarray: Predicted target values.
        """


        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression with Gradient Descent")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for gradient descent")
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of iterations for gradient descent")
    args = parser.parse_args()

    if args.data:
        # Load data
        data = np.genfromtxt(args.data, delimiter=',')
        X = data[1:24, :-1]
        y = data[1:24, -1]
        X_test = data[25:, :-1]
        Y_test=data[25:,-1]
        # Initialize and train the model
        model = LinearRegressionGradientDescent(learning_rate=args.learning_rate,
                                                 n_iterations=args.n_iterations,
                                                )
        model.fit(X, y)


        # Make predictions
        predictions = model.predict(X_test)

        print("Predictions:", predictions)
        
        print()
        print("Accuracy is :",r2_score(list(Y_test),list(predictions)))
    else:
        print("Please provide the path to the data file using the '--data' argument.")
