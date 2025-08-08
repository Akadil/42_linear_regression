import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linear_regression.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

THETA_0 = 0.0
THETA_1 = 0.0
LEARNING_RATE = 0.01
ITERATIONS = 1000
TOLERANCE = 1e-6  # Convergence tolerance

class PredictCarPriceFromMileage:
    """
    A class to predict a car price based on mileage driven. This is a 42 project called 
    ft_linear_regression which serves as a future machine learning projects.

    How the model is trained:
    - The model is trained using a simple linear regression approach.
    - The model parameters (theta_0 and theta_1) are initialized to zero.
    - The learning rate is set to a small value to ensure gradual updates during training.
    - The model is trained using gradient descent to minimize the cost function.
    """
    def __init__(self):
        self.theta_0 = THETA_0
        self.theta_1 = THETA_1
        self.training_history = []  # Store cost history for analysis
        self.is_trained = False

        logger.info("Initialized PredictCarPriceFromMileage model")
        logger.info(f"Initial parameters: theta_0={self.theta_0}, theta_1={self.theta_1}")

    def get_price(self, mileage: float) -> float:
        """
        Predict the price of a car based on its mileage.

        :param mileage: The mileage driven by the car (must be non-negative).
        :return: Predicted price of the car.
        :raises ValueError: If mileage is negative.
        """
        if mileage < 0:
            logger.error(f"Invalid mileage: {mileage}. Mileage cannot be negative.")
            raise ValueError("Mileage cannot be negative")
        
        if not self.is_trained:
            logger.warning("Model has not been trained yet. Using initial parameters.")
        
        predicted_price = self.theta_0 + self.theta_1 * mileage
        logger.debug(f"Predicted price for {mileage} miles: ${predicted_price:.2f}")

        return predicted_price


    def train(self, data: List[Tuple[float, float]], iterations: int = ITERATIONS, 
              learning_rate: float = LEARNING_RATE, verbose: bool = True) -> None:
        """
        Train the model using the provided data.

        :param data: The training data, a list of (mileage, price) tuples.
        :param iterations: The number of training iterations.
        :param learning_rate: The learning rate for gradient descent.
        :param verbose: If True, log training progress every 100 iterations.
        :raises ValueError: If data is empty or contains invalid values.
        """
        if not data:
            logger.error("Training data is empty")
            raise ValueError("Training data cannot be empty")
        
        # Validate data
        self._validate_training_data(data)
        
        logger.info(f"Starting training with {len(data)} data points")
        logger.info(f"Training parameters: iterations={iterations}, learning_rate={learning_rate}")
        
        # Normalize data for better convergence
        normalized_data, mileage_stats = self._normalize_data(data)
        
        prev_cost = float('inf')
        
        for i in range(iterations):
            cost = self._calculate_cost(normalized_data)
            self.training_history.append(cost)
            
            # Log progress
            if verbose and (i == 0 or (i + 1) % 100 == 0 or i == iterations - 1):
                logger.info(f"Iteration {i+1}/{iterations}: Cost = {cost:.6f}, "
                           f"theta_0 = {self.theta_0:.6f}, theta_1 = {self.theta_1:.6f}")
            
            # Check for convergence
            if abs(prev_cost - cost) < TOLERANCE:
                logger.info(f"Converged at iteration {i+1} with cost {cost:.6f}")
                break
            
            self._gradient_descent_step(normalized_data, learning_rate)
            prev_cost = cost
        
        # Denormalize parameters
        self._denormalize_parameters(mileage_stats)
        
        self.is_trained = True
        final_cost = self._calculate_cost(data)
        logger.info(f"Training completed. Final cost: {final_cost:.6f}")
        logger.info(f"Final parameters: theta_0={self.theta_0:.6f}, theta_1={self.theta_1:.6f}")

    def _validate_training_data(self, data: List[Tuple[float, float]]) -> None:
        """Validate that training data contains valid values."""
        for i, (mileage, price) in enumerate(data):
            if not isinstance(mileage, (int, float)) or not isinstance(price, (int, float)):
                raise ValueError(f"Data point {i} contains non-numeric values: {mileage}, {price}")
            if mileage < 0:
                raise ValueError(f"Data point {i} has negative mileage: {mileage}")
            if price <= 0:
                raise ValueError(f"Data point {i} has non-positive price: {price}")
        logger.debug(f"Training data validation passed for {len(data)} data points")

    def _normalize_data(self, data: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], dict]:
        """Normalize mileage data to improve convergence."""
        mileages = [mileage for mileage, _ in data]
        prices = [price for _, price in data]
        
        mileage_mean = np.mean(mileages)
        mileage_std = np.std(mileages)
        price_mean = np.mean(prices)
        
        # Avoid division by zero
        if mileage_std == 0:
            mileage_std = 1
            
        normalized_data = [
            ((mileage - mileage_mean) / mileage_std, price) 
            for mileage, price in data
        ]
        
        mileage_stats = {
            'mean': mileage_mean,
            'std': mileage_std,
            'price_mean': price_mean
        }
        
        logger.debug(f"Data normalized: mileage_mean={mileage_mean:.2f}, mileage_std={mileage_std:.2f}")
        return normalized_data, mileage_stats

    def _denormalize_parameters(self, mileage_stats: dict) -> None:
        """Convert normalized parameters back to original scale."""
        mileage_mean = mileage_stats['mean']
        mileage_std = mileage_stats['std']
        
        # Convert back to original scale
        original_theta_1 = self.theta_1 / mileage_std
        original_theta_0 = self.theta_0 - original_theta_1 * mileage_mean
        
        self.theta_0 = original_theta_0
        self.theta_1 = original_theta_1
        logger.debug("Parameters denormalized to original scale")

    def _calculate_cost(self, data: List[Tuple[float, float]]) -> float:
        """Calculate the mean squared error cost function."""
        m = len(data)
        total_cost = 0.0
        
        for mileage, price in data:
            prediction = self.theta_0 + self.theta_1 * mileage
            total_cost += (prediction - price) ** 2
        
        return total_cost / (2 * m)

    def _gradient_descent_step(self, data: List[Tuple[float, float]], learning_rate: float) -> None:
        """Perform one step of gradient descent."""
        m = len(data)
        sum_errors_0 = 0.0
        sum_errors_1 = 0.0
        
        # Calculate the gradients for theta_0 and theta_1
        for mileage, price in data:
            error = self.theta_0 + self.theta_1 * mileage - price
            sum_errors_0 += error
            sum_errors_1 += error * mileage
        
        # Update the parameters using the calculated gradients
        self.theta_0 -= learning_rate * sum_errors_0 / m
        self.theta_1 -= learning_rate * sum_errors_1 / m

    def load_data_from_csv(self, filepath: str) -> List[Tuple[float, float]]:
        """
        Load training data from a CSV file.
        
        :param filepath: Path to the CSV file with 'km' and 'price' columns.
        :return: List of (mileage, price) tuples.

        :raises FileNotFoundError: If the file doesn't exist.
        :raises ValueError: If the file format is invalid.
        """
        if not os.path.exists(filepath):
            logger.error(f"CSV file not found: {filepath}")
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading data from {filepath}")
            
            # Validate required columns
            if 'km' not in df.columns or 'price' not in df.columns:
                raise ValueError("CSV must contain 'km' and 'price' columns")
            
            # Remove any rows with missing values
            df = df.dropna()
            data = [(float(row['km']), float(row['price'])) for _, row in df.iterrows()]
            
            logger.info(f"Successfully loaded {len(data)} data points from CSV")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model parameters to a file.
        
        :param filepath: Path where to save the model.
        """
        if not self.is_trained:
            logger.warning("Model has not been trained yet")
        
        model_data = {
            'theta_0': self.theta_0,
            'theta_1': self.theta_1,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        :param filepath: Path to the saved model file.
        """
        try:
            import json
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.theta_0 = model_data['theta_0']
            self.theta_1 = model_data['theta_1']
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', [])
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Loaded parameters: theta_0={self.theta_0:.6f}, theta_1={self.theta_1:.6f}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_training_stats(self) -> dict:
        """
        Get statistics about the training process.
        
        :return: Dictionary with training statistics.
        """
        if not self.training_history:
            return {"message": "No training history available"}
        
        return {
            "iterations_completed": len(self.training_history),
            "initial_cost": self.training_history[0],
            "final_cost": self.training_history[-1],
            "cost_reduction": self.training_history[0] - self.training_history[-1],
            "theta_0": self.theta_0,
            "theta_1": self.theta_1,
            "is_trained": self.is_trained
        }


def main():
    """Main function demonstrating the usage of PredictCarPriceFromMileage."""
    logger.info("Starting linear regression demonstration")
    
    # Initialize model
    model = PredictCarPriceFromMileage()
    
    # Try to load data from CSV file, fallback to example data
    try:
        data_path = Path(__file__).parent.parent / "static" / "data.csv"
        if data_path.exists():
            data = model.load_data_from_csv(str(data_path))
        else:
            logger.info("CSV file not found, using example data")
            data = [
                (10000, 15000),
                (20000, 12000),
                (30000, 10000),
                (40000, 8000),
                (50000, 6000)
            ]
    except Exception as e:
        logger.warning(f"Failed to load CSV data: {e}. Using example data.")
        data = [
            (10000, 15000),
            (20000, 12000),
            (30000, 10000),
            (40000, 8000),
            (50000, 6000)
        ]
    
    # Train the model
    try:
        logger.info("Training model...")
        model.train(data, iterations=1000, learning_rate=0.01, verbose=True)
        
        # Display training statistics
        stats = model.get_training_stats()
        logger.info("Training Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Make some predictions
        test_mileages = [25000, 100000, 150000]
        logger.info("Making predictions:")
        
        for mileage in test_mileages:
            try:
                predicted_price = model.get_price(mileage)
                logger.info(f"  Predicted price for {mileage:,} miles: ${predicted_price:,.2f}")
            except ValueError as e:
                logger.error(f"  Error predicting price for {mileage} miles: {e}")
        
        # Save the trained model
        model_path = "trained_model.json"
        try:
            model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    logger.info("Linear regression demonstration completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()