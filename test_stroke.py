import requests
import json

url = "http://localhost:5000/predict-stroke"
data = {
    "gender": "Male",
    "age": 50,
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 120.5,
    "bmi": 28.5,
    "smoking_status": "formerly smoked"
}

headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")