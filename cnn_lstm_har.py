# CNN-LSTM for Human Activity Recognition
# Dataset: Simulated UCI HAR-style (6 activities, 9 sensor channels)
# Framework: TensorFlow / Keras

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

tf.random.set_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
ACTIVITIES  = ['Walking', 'WalkingUpstairs', 'WalkingDownstairs',
               'Sitting', 'Standing', 'Laying']
N_CLASSES   = 6
N_TIMESTEPS = 128    # window length (UCI HAR uses 2.56 s at 50 Hz)
N_CHANNELS  = 9      # acc_x/y/z, gyro_x/y/z, total_acc_x/y/z
N_PER_CLASS = 400
EPOCHS      = 80
BATCH_SIZE  = 64

# ------------------------------------------------------------------
# 1. Dataset generation
# ------------------------------------------------------------------
# Each activity has a distinct dominant frequency and amplitude.
# To simulate inter-subject variability, amplitude and frequency are
# perturbed per sample. A slow drift term models sensor bias.

SIGNAL_PARAMS = {
    0: (1.20, 0.85, 0.12, 0.10),   # Walking
    1: (1.75, 1.00, 0.14, 0.12),   # Walking Upstairs
    2: (1.50, 0.92, 0.14, 0.12),   # Walking Downstairs
    3: (0.28, 0.14, 0.08, 0.05),   # Sitting
    4: (0.22, 0.16, 0.09, 0.06),   # Standing
    5: (0.06, 0.05, 0.03, 0.02),   # Laying
}

def generate_signals(activity_id, n_samples, rng):
    freq, amp, noise_std, drift_std = SIGNAL_PARAMS[activity_id]
    t = np.linspace(0, 6 * np.pi, N_TIMESTEPS)
    X = np.zeros((n_samples, N_TIMESTEPS, N_CHANNELS), dtype=np.float32)

    for s in range(n_samples):
        a = amp  * (1.0 + 0.15 * rng.randn())
        f = freq * (1.0 + 0.10 * rng.randn())
        n = noise_std * (1.0 + 0.30 * abs(rng.randn()))
        phase = rng.uniform(0, 2 * np.pi)
        drift = np.linspace(0, drift_std * rng.randn(), N_TIMESTEPS)

        for ch in range(N_CHANNELS):
            X[s, :, ch] = (
                a        * np.sin(f * t + phase + ch * 0.3) +
                0.20 * a * np.sin(2 * f * t + phase) +
                0.10 * a * np.sin(3 * f * t + phase) +
                drift + n * rng.randn(N_TIMESTEPS)
            ).astype(np.float32)
    return X

rng = np.random.RandomState(42)
X = np.concatenate([generate_signals(i, N_PER_CLASS, rng) for i in range(N_CLASSES)])
y = np.repeat(np.arange(N_CLASSES), N_PER_CLASS).astype(np.int32)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Per-channel z-score normalisation (statistics computed on training set)
for ch in range(N_CHANNELS):
    mu    = X_train[:, :, ch].mean()
    sigma = X_train[:, :, ch].std() + 1e-8
    X_train[:, :, ch] = (X_train[:, :, ch] - mu) / sigma
    X_test[ :, :, ch] = (X_test[ :, :, ch] - mu) / sigma

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ------------------------------------------------------------------
# 2. Model definition
# ------------------------------------------------------------------
# Three Conv1D blocks extract local temporal features, reducing the
# sequence from 128 to 16 steps. The LSTM then models the temporal
# relationships across this compressed feature sequence.

inp = tf.keras.Input(shape=(N_TIMESTEPS, N_CHANNELS))

# CNN block 1
x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
x = tf.keras.layers.Dropout(0.20)(x)

# CNN block 2
x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
x = tf.keras.layers.Dropout(0.20)(x)

# CNN block 3
x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
x = tf.keras.layers.Dropout(0.20)(x)

# LSTM layer
x = tf.keras.layers.LSTM(128, dropout=0.25, recurrent_dropout=0.10)(x)

# Classifier
x   = tf.keras.layers.Dense(64, activation='relu')(x)
x   = tf.keras.layers.Dropout(0.35)(x)
out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inp, out, name='cnn_lstm_har')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)
model.summary()

# ------------------------------------------------------------------
# 3. Training
# ------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-5, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# ------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------
y_pred   = model.predict(X_test, verbose=0).argmax(axis=1)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=ACTIVITIES))

# ------------------------------------------------------------------
# 5. Plots
# ------------------------------------------------------------------
epochs_range = range(1, len(history.history['loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(epochs_range, history.history['loss'],     color='#e74c3c', lw=2, label='Train')
axes[0].plot(epochs_range, history.history['val_loss'], color='#3498db', lw=2, label='Validation')
axes[0].set_title('Loss over Epochs', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, [a * 100 for a in history.history['accuracy']],     color='#2ecc71', lw=2, label='Train')
axes[1].plot(epochs_range, [a * 100 for a in history.history['val_accuracy']], color='#e67e22', lw=2, label='Validation')
axes[1].set_title('Accuracy over Epochs', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(ACTIVITIES, rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(ACTIVITIES, fontsize=9)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: training_curves.png | confusion_matrix.png")
