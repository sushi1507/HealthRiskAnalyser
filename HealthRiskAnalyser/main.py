from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib, json, numpy as np, os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

app = FastAPI(title="Health Risk Analyzer API", version="1.0")

# ✅ Model directory (make sure this path matches your repo structure)
model_dir = os.path.join(os.getcwd(), "mmodels")

# ✅ Load all models
try:
    rf_model = joblib.load(f"{model_dir}/rf_model_20251022_194624.pkl")
    xgb_model = joblib.load(f"{model_dir}/xgb_model_20251022_212451.pkl")
    svm_model = joblib.load(f"{model_dir}/svm_model_20251022_212451.pkl")
    iso_model = joblib.load(f"{model_dir}/iso_model_20251022_194624.pkl")
    autoencoder_model = load_model(f"{model_dir}/autoencoder_model_tf215.h5", compile=False)
    scaler = joblib.load(f"{model_dir}/score_scaler.pkl")

    with open(f"{model_dir}/feature_order.json") as f:
        feature_order = json.load(f)

    print("✅ All models loaded successfully (RF, XGB, SVM, ISO, Autoencoder).")

except Exception as e:
    print(f"❌ Model loading error: {e}")
    feature_order, autoencoder_model = [], None


# 🩺 Root endpoint for health check
@app.get("/")
def root():
    return {"message": "✅ Health Risk Analyzer API active!"}


# 🧠 Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print(f"🧾 Uploaded CSV columns: {list(df.columns)}")

        # Check for missing or extra columns
        missing_cols = [c for c in feature_order if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in feature_order]

        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = 0

        if extra_cols:
            print(f"ℹ️ Ignoring extra columns: {extra_cols}")

        # ✅ Drop target or irrelevant columns
        safe_feature_order = [col for col in feature_order if col != "target"]
        df = df[safe_feature_order]

        # ✅ Predict with each model
        rf_pred = rf_model.predict_proba(df)[:, 1]
        xgb_pred = xgb_model.predict_proba(df)[:, 1]
        svm_pred = svm_model.decision_function(df)
        iso_pred = -iso_model.score_samples(df)

        # ✅ Autoencoder reconstruction
        scaled_data = scaler.transform(df)
        ae_recon = autoencoder_model.predict(scaled_data)
        ae_error = np.mean(np.square(scaled_data - ae_recon), axis=1)

        # ✅ Ensemble score
        ensemble_score = (rf_pred + xgb_pred + svm_pred + iso_pred + ae_error) / 5

        results = [
            {"Patient": int(i + 1), "Risk_Score": float(s), "High_Risk": bool(s > 0.5)}
            for i, s in enumerate(ensemble_score)
        ]

        return {"status": "success", "results": results}

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"status": "error", "message": str(e)}


# 🚀 Local test mode
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
