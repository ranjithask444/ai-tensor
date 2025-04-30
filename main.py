from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# Load the trained Keras model
model = tf.keras.models.load_model("assignment_tf_model.keras")

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Define expected input feature order
feature_columns = [
    'task_priority',
    'deadline_hours',
    'available_bandwidth',
    'skill_match_count',
    'skill_match_percentage'
]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input JSON
        input_data = request.get_json()

        # Validate input
        if not all(col in input_data for col in feature_columns):
            return jsonify({"error": f"Missing one or more required fields: {feature_columns}"}), 400

        # Convert input into feature array
        features = np.array([[input_data[col] for col in feature_columns]], dtype=np.float32)

        # Make prediction
        probabilities = model.predict(features)
        predicted_index = np.argmax(probabilities, axis=1)[0]
        predicted_employee = label_encoder.inverse_transform([predicted_index])[0]

        return jsonify({
            "predicted_employee_id": predicted_employee,
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
        app.run(debug=True,host="0.0.0.0", port=8080)

