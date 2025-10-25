from tensorflow.keras.models import load_model

m = load_model("mmodels/autoencoder_model_tf215.h5")
m.summary()
