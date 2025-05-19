# pipeline.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

# # ── Reproducibility ─────────────────────────────────────────────────────────────
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# --- 1. Configuration ---
DATA_PATH    = "/Users/jd/EcoEdgeAgri/agriproject/data/data.csv"
SEQ_LEN      = 24
BATCH_SIZE   = 32
EPOCHS       = 50
VAL_SPLIT    = 0.2
NOISE_STD    = 0.01
DROPOUT_RATE = 0.05
MODEL_DIR    = "/Users/jd/EcoEdgeAgri/agriproject/models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES   = ["Soil_Moisture", "Ambient_Temperature", "Humidity"]
TARGET_IDX = [0, 1, 2]  # indices within FEATURES

# --- 2. Load & preprocess dataset ---
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df.sort_values(["Plant_ID", "Timestamp"], inplace=True)

scaled = []
scalers = {}
for pid, sub in df.groupby("Plant_ID"):
    sc = MinMaxScaler()
    sub_scaled = sub.copy()
    sub_scaled[FEATURES] = sc.fit_transform(sub[FEATURES])
    scaled.append(sub_scaled)
    scalers[pid] = sc

df_norm = pd.concat(scaled).sort_values(["Plant_ID", "Timestamp"]).reset_index(drop=True)

# --- 3. Build sliding-window sequences ---
X_all, y_all = [], []
for pid, sub in df_norm.groupby("Plant_ID"):
    vals = sub[FEATURES].values
    for i in range(len(vals) - SEQ_LEN):
        X_all.append(vals[i : i + SEQ_LEN])
        y_all.append(vals[i + SEQ_LEN, TARGET_IDX])

X_all = np.array(X_all, dtype="float32")
y_all = np.array(y_all, dtype="float32")

# train/val split (time-order)
n_val     = int(len(X_all) * VAL_SPLIT)
X_train   = X_all[:-n_val]
y_train   = y_all[:-n_val]
X_val     = X_all[-n_val:]
y_val     = y_all[-n_val:]

# --- 4. Augment training data ---
def augment(X, noise_std, dropout_rate):
    Xn = X + np.random.normal(0, noise_std, X.shape)
    Xn = np.clip(Xn, 0.0, 1.0)
    if dropout_rate > 0:
        mask = (np.random.rand(*Xn.shape) < dropout_rate)
        Xn[mask] = 0.0
    return Xn

X_aug = augment(X_train, NOISE_STD, DROPOUT_RATE)
y_aug = y_train.copy()

X_train = np.concatenate([X_train, X_aug], axis=0)
y_train = np.concatenate([y_train, y_aug], axis=0)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# --- 5. Build the TCN model ---
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)
    c1 = layers.Conv1D(filters, kernel_size, padding="causal",
                       dilation_rate=dilation_rate, activation="relu")(x)
    d1 = layers.Dropout(dropout_rate)(c1)
    c2 = layers.Conv1D(filters, kernel_size, padding="causal",
                       dilation_rate=dilation_rate, activation="relu")(d1)
    d2 = layers.Dropout(dropout_rate)(c2)
    return layers.Add()([x, d2])

inp = keras.Input(shape=(SEQ_LEN, len(FEATURES)))
x = inp
for d in [1, 2, 4]:
    x = tcn_block(x, filters=32, kernel_size=3, dilation_rate=d, dropout_rate=0.1)
x = layers.Lambda(lambda t: t[:, -1, :])(x)  # take last time-step
out = layers.Dense(len(FEATURES), activation="linear")(x)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# --- 6. Train with Early Stopping ---
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)

# --- Plot Training Curves ---
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "training_loss.png"))

plt.figure()
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.title("MAE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "training_mae.png"))

# --- 7. Evaluate on validation set ---
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2  = r2_score(y_val, y_pred)

print("\n=== Evaluation on Validation Data ===")
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# --- Scatter Plots: Actual vs. Predicted ---
for i, feat in enumerate(FEATURES):
    plt.figure(figsize=(5,4))
    plt.scatter(y_val[:, i], y_pred[:, i], alpha=0.5, s=10)
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.title(f"{feat}: Actual vs. Predicted")
    plt.xlabel("Actual (normalized)")
    plt.ylabel("Predicted (normalized)")
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, f"scatter_{feat}.png"))

# --- Time-Series Comparison for first N samples ---
N = 3
for j in range(N):
    plt.figure(figsize=(6,3))
    plt.plot(y_val[j], label="Actual")
    plt.plot(y_pred[j], '--', label="Predicted")
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=45)
    plt.title(f"Sample #{j+1}: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"timeseries_sample{j+1}.png"))

    # --- Plot Training Curves ---
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "training_loss.png"))
    plt.show()      # <-- display the plot interactively

    plt.figure()
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.title("MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "training_mae.png"))
    plt.show()      # <-- display this one too


# --- 8. Save Keras model ---
h5_path = os.path.join(MODEL_DIR, "tcn_model.h5")
model.save(h5_path)
print(f"Keras model saved to: {h5_path}")

# --- 9. Convert to quantized TFLite ---
def representative_gen():
    for i in range(min(100, len(X_train))):
        yield [X_train[i : i + 1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = os.path.join(MODEL_DIR, "tcn_model_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Quantized TFLite model saved to: {tflite_path}")
print("\n✅ Pipeline complete. Plots saved in:", MODEL_DIR)
