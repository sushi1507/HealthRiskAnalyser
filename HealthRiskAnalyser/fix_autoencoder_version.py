from tensorflow.keras.models import load_model

# Load your existing model
model = load_model("mmodels/autoencoder_model_20251022_194624.h5")

# Re-save it in TensorFlow 2.15 compatible format
model.save("mmodels/autoencoder_model_compatible.h5", save_format="h5")

print("✅ Autoencoder successfully re-saved for TensorFlow 2.15 compatibility!")
