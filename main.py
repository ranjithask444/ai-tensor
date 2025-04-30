from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load model and label classes
model = tf.keras.models.load_model("best_employee_model.h5")
label_classes = np.load("label_classes.npy", allow_pickle=True)

app = Flask(__name__)

# Constants
NUM_SKILLS = 10
SKILL_IDS = [str(i) for i in range(1, NUM_SKILLS + 1)]

def one_hot_encode(skills, all_skills=SKILL_IDS):
    return [1 if s in skills else 0 for s in all_skills]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    task_features = [
        data["priority"],
        data["deadline_hours"],
        data["available_bandwidth"]
    ]
    task_skill_vec = one_hot_encode(data["skills_required"])

    candidates = data["candidate_employees"]
    rows = []
    user_ids = []

    for emp in candidates:
        emp_skill_vec = one_hot_encode(emp["skills"])
        num_matching = sum([1 for s in data["skills_required"] if s in emp["skills"]])

        if num_matching == 0:
            continue

        match_score = num_matching / len(data["skills_required"])

        row = (
            task_features +
            task_skill_vec +
            [emp["available_bandwidth"]] +
            emp_skill_vec +
            [match_score, num_matching]
        )
        rows.append(row)
        user_ids.append(emp["user_id"])

    if not rows:
        return jsonify({"selected_user_id": None, "reason": "No candidates with matching skills"})

    predictions = model.predict(np.array(rows)).flatten()
    best_index = np.argmax(predictions)
    best_user_id = user_ids[best_index]

    return jsonify({"selected_user_id": best_user_id})


@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
        app.run(debug=True,host="0.0.0.0", port=8080)

