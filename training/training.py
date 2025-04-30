import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("employee_task_training_data.csv")

# Encode label_user_id to numeric values
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label_user_id"])

# Drop non-feature columns
X = df.drop(columns=["label_user_id", "label"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Single score output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Use 'match_score' as the label
y = df["match_score"]

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)

# Save model and label encoder classes
model.save("best_employee_model.h5")
np.save("label_classes.npy", label_encoder.classes_)
