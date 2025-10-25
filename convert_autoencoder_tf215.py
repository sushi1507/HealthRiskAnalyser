import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
import h5py

print("✅ TensorFlow version:", tf.__version__)

try:
    print("📦 Trying to open model file safely...")
    with h5py.File("mmodels/autoencoder_model_tf215.h5", "r") as f:
        print("✅ H5 file structure opened successfully.")
        if "model_weights" not in f:
            raise ValueError("❌ No model_weights found inside this file.")

        # Handle both byte and string encodings
        raw_names = f["model_weights"].attrs["layer_names"]
        layer_names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in raw_names
        ]
        print(f"📋 Layers found: {layer_names}")

    # Define a new clean model manually
    input_dim = 25  # change this if your dataset has a different feature count
    model = Sequential([
        Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(input_dim, activation="sigmoid")
    ])

    model.save("mmodels/autoencoder_model_compatible.h5", include_optimizer=False)
    print("✅ Saved new clean model as autoencoder_model_compatible.h5")

except Exception as e:
    print(f"❌ Conversion failed: {e}")
