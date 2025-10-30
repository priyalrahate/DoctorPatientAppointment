import sys
import os
import pandas as pd

# Add the ml-backend directory to the path
sys.path.append(os.path.abspath('.'))

from src.stroke.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

def test_stroke_prediction():
    try:
        print("Creating custom data...")
        # Create test data
        custom_data = CustomData(
            gender="Male",
            age=50,
            hypertension=1,
            heart_disease=0,
            ever_married="Yes",
            work_type="Private",
            Residence_type="Urban",
            avg_glucose_level=120.5,
            bmi=28.5,
            smoking_status="formerly smoked"
        )
        
        print("Converting to DataFrame...")
        input_df = custom_data.get_data_as_data_frame()
        print(f"Input DataFrame:\n{input_df}")
        
        print("Creating prediction pipeline...")
        pipe = PredictPipeline()
        
        print("Making prediction...")
        preds = pipe.predict(input_df)
        print(f"Predictions: {preds}")
        
        result = "Stroke" if int(preds[0]) == 1 else "No Stroke"
        print(f"Result: {result}")
        
    except CustomException as e:
        print(f"Custom Exception: {e}")
        print(f"Error message: {e.error_message}")
    except Exception as e:
        print(f"General Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stroke_prediction()