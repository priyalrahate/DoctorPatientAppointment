import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

class MockModel:
    """
    A mock model that returns a fixed prediction for stroke risk.
    This is a temporary solution to get the system running while
    we work on retraining the model with the current version of scikit-learn.
    """
    
    def predict(self, features):
        """
        Mock prediction function that returns a random prediction
        based on some simple heuristics.
        
        Args:
            features: Input features as a DataFrame
            
        Returns:
            Array with prediction (0 for no stroke, 1 for stroke)
        """
        try:
            # Simple heuristic-based prediction
            # Higher age, hypertension, and heart disease increase risk
            age = features['age'].iloc[0]
            hypertension = features['hypertension'].iloc[0]
            heart_disease = features['heart_disease'].iloc[0]
            bmi = features['bmi'].iloc[0]
            avg_glucose_level = features['avg_glucose_level'].iloc[0]
            
            # Calculate risk score based on known risk factors
            risk_score = 0
            
            # Age factor (risk increases significantly after 55)
            if age > 55:
                risk_score += 2
            elif age > 40:
                risk_score += 1
                
            # Medical conditions
            risk_score += hypertension * 3
            risk_score += heart_disease * 4
            
            # BMI factor (obesity increases risk)
            if bmi > 30:
                risk_score += 2
            elif bmi > 25:
                risk_score += 1
                
            # Glucose level factor (diabetes increases risk)
            if avg_glucose_level > 126:  # Diabetic range
                risk_score += 3
            elif avg_glucose_level > 100:  # Pre-diabetic range
                risk_score += 1
                
            # Smoking (significant risk factor)
            smoking_status = features['smoking_status'].iloc[0]
            if smoking_status in ['smokes', 'formerly smoked']:
                risk_score += 2
                
            # Work type (high stress jobs may increase risk)
            work_type = features['work_type'].iloc[0]
            if work_type == 'Private':
                risk_score += 1
                
            # Random factor to add some variability
            random_factor = np.random.randint(0, 3)
            risk_score += random_factor
            
            # Convert risk score to prediction (higher scores = higher risk)
            # Threshold is set to give roughly 5% positive predictions
            prediction = 1 if risk_score >= 8 else 0
            
            return np.array([prediction])
            
        except Exception as e:
            raise CustomException(e, sys)

class MockPreprocessor:
    """
    A mock preprocessor that simply passes through the data.
    """
    
    def transform(self, features):
        """
        Mock transform function that returns the features as-is.
        
        Args:
            features: Input features as a DataFrame
            
        Returns:
            The same features (no transformation)
        """
        return features

def load_mock_objects():
    """
    Function to load mock model and preprocessor objects.
    
    Returns:
        Tuple of (mock_model, mock_preprocessor)
    """
    mock_model = MockModel()
    mock_preprocessor = MockPreprocessor()
    return mock_model, mock_preprocessor