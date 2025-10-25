from tensorflow.keras.models import load_model, save_model

# Load your current model trained on higher TF version
model = load_model("mmodels/autoencoder_model_compatible.h5", compile=False)

# Save it again in a TF 2.15-compatible format
save_model(model, "mmodels/autoencoder_model_tf215.h5")
print("✅ Model successfully re-saved for TensorFlow 2.15 compatibility.")
