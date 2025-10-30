from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
# Use the fixed pipeline instead of the original one
from src.stroke.predict_pipeline_fixed import PredictPipeline as StrokePredictPipeline, CustomData as StrokeCustomData
from src.diabetes.predict_pipeline import PredictPipeline as DiabetesPredictPipeline, CustomData as DiabetesCustomData
from src.diseases_and_symptoms.predict_pipeline import PredictPipeline as DiseasePredictPipeline

app = Flask(__name__)
CORS(app)

# Route for predicting stroke
@app.route('/predict-stroke', methods=['POST'])
def predict_stroke():
    try:
        data = request.get_json()

        # Validate input data
        required_fields = [
            "gender", "age", "hypertension", "heart_disease", 
            "ever_married", "work_type", "Residence_type", 
            "avg_glucose_level", "bmi", "smoking_status"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        custom_data = StrokeCustomData(**data)
        input_df = custom_data.get_data_as_data_frame()
        pipe = StrokePredictPipeline()
        preds = pipe.predict(input_df)

        return jsonify({"prediction": "Stroke" if preds[0] == 1 else "No Stroke"})
    except Exception as e:
        app.logger.error(f"Error in /predict-stroke: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Route for predicting diabetes
@app.route('/predict-diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()

        # Validate input data
        required_fields = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
            
        custom_data = DiabetesCustomData(**data)
        input_df = custom_data.get_data_as_data_frame()
        pipe = DiabetesPredictPipeline()
        preds = pipe.predict(input_df)

        return jsonify({"prediction": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for predicting diseases based on symptoms
@app.route('/predict-disease-using-symptoms', methods=['POST'])
def predict_disease():
    try:
        data = request.get_json()

        # Validate input data
        if 'symptoms' not in data:
            return jsonify({"error": "Missing field: symptoms"}), 400
        
        symptoms_list = data['symptoms']

        # Validate the number of symptoms
        if not isinstance(symptoms_list, list) or len(symptoms_list) < 3 or len(symptoms_list) > 5:
            return jsonify({"error": "Provide between 3 and 5 symptoms."}), 400

        pipe = DiseasePredictPipeline()
        predicted_disease = pipe.predict(symptoms_list)

        return jsonify({"prediction": predicted_disease})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides a port
    app.run(host="0.0.0.0", port=port, debug=True)


# # app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import traceback

# # Import your project pipelines
# from src.stroke.predict_pipeline import PredictPipeline as StrokePredictPipeline, CustomData as StrokeCustomData
# from src.diabetes.predict_pipeline import PredictPipeline as DiabetesPredictPipeline, CustomData as DiabetesCustomData
# from src.diseases_and_symptoms.predict_pipeline import PredictPipeline as DiseasePredictPipeline

# app = Flask(__name__)
# # Allow CORS for all origins during development. In production, restrict origins.
# CORS(app, resources={r"/*": {"origins": "*"}})


# @app.route("/", methods=["GET"])
# @app.route("", methods=["GET"])
# def home():
#     return jsonify({"status": "OK", "message": "API is running"}), 200


# # Accept both with and without trailing slash
# @app.route("/predict-stroke", methods=["POST"])
# @app.route("/predict-stroke/", methods=["POST"])
# def predict_stroke():
#     try:
#         data = request.get_json(force=True, silent=False)
#         if not data:
#             return jsonify({"error": "No JSON payload received"}), 400

#         required_fields = [
#             "gender", "age", "hypertension", "heart_disease",
#             "ever_married", "work_type", "Residence_type",
#             "avg_glucose_level", "bmi", "smoking_status"
#         ]
#         missing = [f for f in required_fields if f not in data]
#         if missing:
#             return jsonify({"error": "Missing fields", "missing": missing}), 400

#         custom_data = StrokeCustomData(**data)
#         input_df = custom_data.get_data_as_data_frame()
#         pipe = StrokePredictPipeline()
#         preds = pipe.predict(input_df)

#         result = "Stroke" if int(preds[0]) == 1 else "No Stroke"
#         return jsonify({"prediction": result}), 200
#     except Exception as e:
#         # Detailed error for debugging. Remove stacktrace in production.
#         app.logger.error("Error in /predict-stroke:\n" + traceback.format_exc())
#         return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


# @app.route("/predict-diabetes", methods=["POST"])
# @app.route("/predict-diabetes/", methods=["POST"])
# def predict_diabetes():
#     try:
#         data = request.get_json(force=True, silent=False)
#         if not data:
#             return jsonify({"error": "No JSON payload received"}), 400

#         required_fields = [
#             "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
#             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
#         ]
#         missing = [f for f in required_fields if f not in data]
#         if missing:
#             return jsonify({"error": "Missing fields", "missing": missing}), 400

#         custom_data = DiabetesCustomData(**data)
#         input_df = custom_data.get_data_as_data_frame()
#         pipe = DiabetesPredictPipeline()
#         preds = pipe.predict(input_df)

#         # assume preds is numpy array or list
#         return jsonify({"prediction": preds.tolist()}), 200
#     except Exception as e:
#         app.logger.error("Error in /predict-diabetes:\n" + traceback.format_exc())
#         return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


# @app.route("/predict-disease-using-symptoms", methods=["POST"])
# @app.route("/predict-disease-using-symptoms/", methods=["POST"])
# def predict_disease():
#     try:
#         data = request.get_json(force=True, silent=False)
#         if not data:
#             return jsonify({"error": "No JSON payload received"}), 400

#         if 'symptoms' not in data:
#             return jsonify({"error": "Missing field: symptoms"}), 400

#         symptoms_list = data['symptoms']
#         if not isinstance(symptoms_list, list) or not (3 <= len(symptoms_list) <= 5):
#             return jsonify({"error": "Provide between 3 and 5 symptoms."}), 400

#         pipe = DiseasePredictPipeline()
#         predicted_disease = pipe.predict(symptoms_list)

#         return jsonify({"prediction": predicted_disease}), 200
#     except Exception as e:
#         app.logger.error("Error in /predict-disease-using-symptoms:\n" + traceback.format_exc())
#         return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     # debug=True prints full tracebacks. Disable in production.
#     app.run(host="0.0.0.0", port=port, debug=True)
