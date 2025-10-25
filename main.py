from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib, json, numpy as np, os, requests
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

app = FastAPI(title="Health Risk Analyzer API", version="1.0")

# ✅ Model directory (create if missing)
model_dir = os.path.join(os.getcwd(), "mmodels")
os.makedirs(model_dir, exist_ok=True)

# ✅ Direct Google Drive download links (uc?id format)
model_urls = {
    "rf_model_20251022_194624.pkl": "https://drive.google.com/uc?id=1vdYZCmWM1-IYhDqjt3LLfd25Xsl8eRHV",
    "xgb_model_20251022_212451.pkl": "https://drive.google.com/uc?id=1SYhCbOkbR6qjHJ9kMYJwqeXh27lBnuFp",
    "svm_model_20251022_212451.pkl": "https://drive.google.com/uc?id=15CgVxXcQzw_8any7CXjPzMSZzrC7RZNW",
    "iso_model_20251022_194624.pkl": "https://drive.google.com/uc?id=1bOG11svcEN0FD7Bj-WuFr3HDmK9-tRFX",
    "autoencoder_model_compatible.h5": "https://drive.google.com/uc?id=1dArJ0cLPYP2DneEYdXZ6G5axb4XWFs7Q",
    "score_scaler.pkl": "https://drive.google.com/uc?id=1iY66eGcwVF-q6ogV91xBjB8Y9-yf-R4Y",
    "feature_order.json": "https://drive.google.com/uc?id=1a-giHwjPyXTY3UIPLBxqJy3bwV6hF1Ui"
}

# ✅ Download missing models
for filename, url in model_urls.items():
    local_path = os.path.join(model_dir, filename)
    if not os.path.exists(local_path):
        try:
            print(f"⬇️ Downloading {filename} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"⚠️ Failed to download {filename}: {e}")

# ✅ Load models safely
try:
    rf_model = joblib.load(f"{model_dir}/rf_model_20251022_194624.pkl")
    xgb_model = joblib.load(f"{model_dir}/xgb_model_20251022_212451.pkl")
    svm_model = joblib.load(f"{model_dir}/svm_model_20251022_212451.pkl")
    iso_model = joblib.load(f"{model_dir}/iso_model_20251022_194624.pkl")
    autoencoder_model = load_model(f"{model_dir}/autoencoder_model_compatible.h5", compile=False)
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
        import io

        # ✅ Read uploaded CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        print(f"🧾 Uploaded CSV columns: {list(df.columns)}")

        if df.empty:
            return {"status": "error", "message": "Uploaded CSV is empty or invalid."}

        # ✅ Clean up column names (remove spaces and hidden chars)
        df.columns = [c.strip() for c in df.columns]

        # ✅ Align features with model’s expected order
        safe_feature_order = [col for col in feature_order if col != "target"]

        print("📋 Feature order from model:", safe_feature_order)
        print("📋 Columns in uploaded CSV:", list(df.columns))

        # Add missing columns (fill as 0)
        for col in safe_feature_order:
            if col not in df.columns:
                df[col] = 0

        # Filter only valid model features
        df = df[[col for col in df.columns if col in safe_feature_order]]

        # ✅ Validate feature shape
        if df.shape[1] == 0:
            print("❌ No matching features found between CSV and model feature_order.")
            return {
                "status": "error",
                "message": "No matching features between uploaded CSV and model configuration."
            }

        # ✅ Convert all to numeric safely
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # ✅ Convert to NumPy for model compatibility
        X = df.to_numpy().astype(float)
        print(f"✅ Prepared data shape: {X.shape}")

        # 🧠 Run all model predictions
        rf_pred = rf_model.predict_proba(X)[:, 1]
        xgb_pred = xgb_model.predict_proba(X)[:, 1]
        svm_pred = svm_model.decision_function(X)
        iso_pred = -iso_model.score_samples(X)

        # ✅ Autoencoder
        scaled_data = scaler.transform(X)
        ae_recon = autoencoder_model.predict(scaled_data)
        ae_error = np.mean(np.square(scaled_data - ae_recon), axis=1)

        # ✅ Ensemble prediction
        ensemble_score = (rf_pred + xgb_pred + svm_pred + iso_pred + ae_error) / 5

        results = [
            {"Patient": int(i + 1), "Risk_Score": float(s), "High_Risk": bool(s > 0.5)}
            for i, s in enumerate(ensemble_score)
        ]

        return {"status": "success", "results": results}

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
