import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# 1️⃣ Load the modern model (TF 2.17+ / Keras 3.x)
old_model = load_model("mmodels/autoencoder_model_compatible.h5", compile=False)
print("✅ Loaded original model successfully.")

# 2️⃣ Recreate the same architecture in TF 2.17 code
# Adjust input shape to match your old model (look at summary)
input_dim = old_model.input_shape[1]
inputs = Input(shape=(input_dim,))
x = old_model.layers[1](inputs)
for layer in old_model.layers[2:]:
    x = layer(x)
new_model = Model(inputs, x)
print("✅ Reconstructed architecture.")

# 3️⃣ Copy weights
new_model.set_weights(old_model.get_weights())
print("✅ Copied weights.")

# 4️⃣ Save in legacy H5 format (TF 2.15-compatible)
new_model.save("mmodels/autoencoder_model_tf215.h5", include_optimizer=False, save_format="h5")
print("🎉 Saved new TF 2.15-compatible model.")
