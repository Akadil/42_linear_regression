import pandas as pd

class GradientDescentPricer:
    """
    A class to perform gradient descent optimization.
    """
    def __init__(self, theta_0 = 0.0, theta_1 = 0.0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initializes the GradientDescent optimizer.

        Parameters:
        learning_rate (float): The step size for each iteration.
        max_iterations (int): The maximum number of iterations to perform.
        tolerance (float): The tolerance for convergence.
        theta_0 (float): Initial value for parameter theta_0.
        theta_1 (float): Initial value for parameter theta_1.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.theta_0 = theta_0
        self.theta_1 = theta_1
    

    def estimate_price(self, km):
        """
        Estimates the price based on km using the linear model.
        """
        return self.theta_0 + self.theta_1 * km
    

    def iterate(self, data: pd.DataFrame):
        """
        Performs a single iteration of gradient descent.

        Parameters:
        data (pd.DataFrame): A DataFrame containing 'km' and 'price' columns.
        """
        m = len(data)
        predictions = self.estimate_price(data['km'])
        errors = predictions - data['price']
        
        gradient_0 = (1/m) * errors.sum()
        gradient_1 = (1/m) * (errors * data['km']).sum()
        
        self.theta_0 -= self.learning_rate * gradient_0
        self.theta_1 -= self.learning_rate * gradient_1


    def fit(self, data: pd.DataFrame):
        """
        Fits the model to the data using gradient descent.

        Parameters:
        data (pd.DataFrame): A DataFrame containing 'km' and 'price' columns.
        """
        data_normalized = self.normalize_data(data)
        m = len(data)
        
        for _iteration in range(self.max_iterations):
            predictions = self.estimate_price(data_normalized['km'])
            errors = predictions - data_normalized['price']
            
            gradient_0 = (1/m) * errors.sum()
            gradient_1 = (1/m) * (errors * data_normalized['km']).sum()
            
            new_theta_0 = self.theta_0 - self.learning_rate * gradient_0
            new_theta_1 = self.theta_1 - self.learning_rate * gradient_1
            
            # Check for convergence
            if abs(new_theta_0 - self.theta_0) < self.tolerance and abs(new_theta_1 - self.theta_1) < self.tolerance:
                break
            
            self.theta_0 = new_theta_0
            self.theta_1 = new_theta_1
            
            if _iteration % 10 == 0:  # Print less frequently
                cost = (1/(2*m)) * (errors ** 2).sum()
                print(f"Iteration {_iteration}: cost = {cost:.6f}")

        # DENORMALIZE the parameters
        self.theta_1 = self.theta_1 * (data['price'].std() / data['km'].std())
        self.theta_0 = data['price'].mean() + self.theta_0 * data['price'].std() - self.theta_1 * data['km'].mean()


    def cost_function(self, data: pd.DataFrame):
        """
        Computes the Mean Squared Error cost function.

        Parameters:
        data (pd.DataFrame): A DataFrame containing 'km' and 'price' columns.

        Returns:
        float: The Mean Squared Error.
        """
        m = len(data)
        predictions = self.estimate_price(data['km'])
        errors = predictions - data['price']
        mse = (1/(2*m)) * (errors ** 2).sum()
        return mse
    

    def normalize_data(self, data: pd.DataFrame):
        """
        Normalizes the 'km' and 'price' columns in the DataFrame.

        Parameters:
        data (pd.DataFrame): A DataFrame containing 'km' and 'price' columns.

        Returns:
        pd.DataFrame: A DataFrame with normalized 'km' and 'price'.
        """
        data_normalized = data.copy()
        data_normalized['km'] = (data['km'] - data['km'].mean()) / data['km'].std()
        data_normalized['price'] = (data['price'] - data['price'].mean()) / data['price'].std()
        return data_normalized


    # =============================================================================
    # Getters
    # =============================================================================
    def get_parameters(self):
        """
        Returns the current parameters of the model.

        Returns:
        tuple: A tuple containing (theta_0, theta_1).
        """
        return self.theta_0, self.theta_1


if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("data.csv")


    pricer = GradientDescentPricer(theta_0=0.0, theta_1=0.0)
    pricer.fit(data)
    theta_0, theta_1 = pricer.get_parameters()
    
    # draw the final result.
    import matplotlib.pyplot as plt
    # data_normalized = pricer.normalize_data(data)
    plt.scatter(data['km'], data['price'], color='blue', label='Data points')
    ## draw the fitted line
    x_vals = pd.Series([data['km'].min(), data['km'].max()])
    y_vals = theta_0 + theta_1 * x_vals
    plt.plot(x_vals, y_vals, color='red', label='Fitted line')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.title('Gradient Descent Linear Regression')
    plt.legend()
    plt.show()
    
    
    print(f"Fitted parameters: theta_0 = {theta_0}, theta_1 = {theta_1}")
    print(f"Cost function value: {pricer.cost_function(data)}")