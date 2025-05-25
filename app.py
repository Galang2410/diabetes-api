from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Cek jika file belum ada, download dari Google Drive
if not os.path.exists("diabetes_model_tuned.pkl"):
    import gdown
    url = "https://drive.google.com/uc?id=1dp4n5dim6hnhZvE7YAkLflGFeZaqLUCI"
    gdown.download(url, "diabetes_model_tuned.pkl", quiet=False)

# Load model
model = joblib.load("diabetes_model_tuned.pkl")

# Buat app flask
app = Flask(__name__)

@app.route('/')
def home():
    return "API is running"

@app.route('/predict-form', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Ambil input dan masukkan ke DataFrame
        input_data = pd.DataFrame([data])

        # Prediksi
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

