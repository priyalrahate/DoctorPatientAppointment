import sys
import os
import dill

def check_model():
    try:
        # Change to ml-backend directory
        os.chdir("ml-backend")
        
        model_path = os.path.join("artifacts", "stroke_model.pkl")
        print(f"Checking model file: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print("Attempting to load model...")
            with open(model_path, "rb") as file_obj:
                model = dill.load(file_obj)
            print(f"Model loaded successfully: {type(model)}")
            print(f"Model attributes: {dir(model)}")
        else:
            print("Model file not found!")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model()