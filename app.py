from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import jwt
import psycopg2
from datetime import datetime
import uuid
from functools import wraps

# Database configuration
DATABASE_URL = "postgres://avnadmin:AVNS_N3xy8A8FWn_K_HQY_r8@pg-e296f5a-khaerulilman10-ebe5.b.aivencloud.com:13597/defaultdb?sslmode=require"
JWT_SECRET = "qwdb78qdhqiwanqudib8qudj2ibiug3489fh"

# Cek jika file belum ada, download dari Google Drive
if not os.path.exists("diabetes_model_tuned2.pkl"):
    import gdown
    url = "https://drive.google.com/uc?id=1Zo-xI1y3Y64HU6aiwXVUIHXfu84bBphx"
    gdown.download(url, "diabetes_model_tuned2.pkl", quiet=False)

# Load model
model = joblib.load("diabetes_model_tuned2.pkl")

# Buat app flask
app = Flask(__name__)
CORS(app, origins="*", 
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

def get_db_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token format invalid'}), 401
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            if 'userId' not in data:
                return jsonify({'message': 'Token payload invalid - missing userId'}), 401
            current_user_id = data['userId']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user_id, *args, **kwargs)
    return decorated

@app.route('/')
def home():
    return "API is running"

@app.route('/predict-history', methods=['POST'])
@token_required
def predict_history(current_user_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = [
            'hypertension', 'heart_disease', 'bmi', 'blood_glucose_level',
            'HbA1c_level', 'smoking_history', 'gender', 'age'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = pd.DataFrame([{
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'bmi': float(data['bmi']),
            'blood_glucose_level': float(data['blood_glucose_level']),
            'HbA1c_level': float(data['HbA1c_level']),
            'smoking_history': data['smoking_history'],
            'gender': data['gender'],
            'age': float(data['age'])
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = float(proba[1]) if prediction == 1 else float(proba[0])
        prediction_result = "positive" if prediction == 1 else "negative"

        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500

        try:
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO form_check_history 
                (id, "userId", hypertension, "heartDisease", bmi, "bloodGlucoseLevel", 
                 "hba1cLevel", "smokingHistory", gender, age, "predictionResult", "createdAt", "updatedAt") 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            record_id = str(uuid.uuid4())
            current_time = datetime.now()

            cursor.execute(insert_query, (
                record_id,
                current_user_id,
                bool(int(data['hypertension'])),
                bool(int(data['heart_disease'])),
                float(data['bmi']),
                float(data['blood_glucose_level']),
                float(data['HbA1c_level']),
                data['smoking_history'],
                data['gender'],
                float(data['age']),
                prediction_result,
                current_time,
                current_time
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({
                'success': True,
                'message': 'Prediction completed and saved successfully',
                'userId': current_user_id,
                'recordId': record_id,
                'prediction': {
                    'result': int(prediction),
                    'resultText': prediction_result,
                    'confidence': round(confidence, 4)
                },
                'inputData': data,
                'timestamp': current_time.isoformat()
            }), 200

        except Exception as db_error:
            conn.rollback()
            cursor.close()
            conn.close()
            return jsonify({'error': f'Database error: {str(db_error)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
