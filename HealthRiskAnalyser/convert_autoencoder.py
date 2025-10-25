from tensorflow.keras.models import load_model

# Load the model saved in newer TF
model = load_model("mmodels/autoencoder_model_20251022_194624.h5", compile=False)

# Re-save using legacy HDF5 format (compatible with TF 2.15)
model.save("mmodels/autoencoder_model_compatible.h5", save_format="h5")

print("✅ Model successfully converted to TF 2.15 compatible format.")
