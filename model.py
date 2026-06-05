import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

class CongestionMLFramework:
    """Encapsulates hyperparameter structures and inference logic."""
    def __init__(self, estimators=75):
        self.regressor = RandomForestRegressor(n_estimators=estimators, max_depth=12, random_state=42)
        logging.info(f"Model instanced with {estimators} estimators and depth limits set.")

    def train_predictive_pipeline(self, features, labels):
        """Executes validation distributions and model alignment training."""
        logging.info("Splitting dataset into train-test distributions...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        logging.info("Training predictive Random Forest Regressor engine...")
        self.regressor.fit(X_train, y_train)
        
        train_score = self.regressor.score(X_train, y_train)
        test_score = self.regressor.score(X_test, y_test)
        logging.info(f"Model tracking scores established -> Train R^2: {train_score:.4f} | Test R^2: {test_score:.4f}")

    def compute_live_inference(self, live_array: np.ndarray) -> np.ndarray:
        """Infers real-time optimized window boundaries for outgoing lines."""
        if not hasattr(self.regressor, "estimators_"):
            raise RuntimeError("Cannot execute inference on unaligned weights or untrained models.")
        return self.regressor.predict(live_array)
