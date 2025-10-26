from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib, json, numpy as np, os, requests
from tensorflow.keras.models import load_model

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
    "score_scaler.pkl": "https://drive.google.com/uc?id=1iY66eGcwVF-q6ogV91xBjB8Y9-yf-R4Y"
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

    print("✅ All models loaded successfully (RF, XGB, SVM, ISO, Autoencoder).")

except Exception as e:
    print(f"❌ Model loading error: {e}")
    rf_model = xgb_model = svm_model = iso_model = autoencoder_model = scaler = None

# ✅ Hardcoded feature order (no JSON dependency)
feature_order = [
    "age",
    "resting_blood_pressure",
    "cholestoral",
    "Max_heart_rate",
    "oldpeak",
    "sex_Male",
    "chest_pain_type_Atypical angina",
    "chest_pain_type_Non-anginal pain",
    "chest_pain_type_Typical angina",
    "fasting_blood_sugar_Lower than 120 mg/ml",
    "rest_ecg_Normal",
    "rest_ecg_ST-T wave abnormality",
    "exercise_induced_angina_Yes",
    "slope_Flat",
    "slope_Upsloping",
    "vessels_colored_by_flourosopy_One",
    "vessels_colored_by_flourosopy_Three",
    "vessels_colored_by_flourosopy_Two",
    "vessels_colored_by_flourosopy_Zero",
    "thalassemia_No",
    "thalassemia_Normal",
    "thalassemia_Reversable Defect",
    "iso_score",
    "auto_score"
]

print(f"✅ Loaded {len(feature_order)} features directly into memory.")

# 🩺 Root endpoint
@app.get("/")
def root():
    return {"message": "✅ Health Risk Analyzer API active!"}

# 🧠 Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        import io

        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        print(f"🧾 Uploaded CSV columns: {list(df.columns)}")

        if df.empty:
            return {"status": "error", "message": "Uploaded CSV is empty or invalid."}

        df.columns = [c.strip() for c in df.columns]

        # ✅ Add missing columns
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0

        # ✅ Keep only expected features in exact order
        df = df[feature_order]

        print(f"✅ Prepared feature-aligned data shape: {df.shape}")

        # ✅ Convert all to numeric
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        X = df.to_numpy().astype(float)

        # 🧠 Run models
        rf_pred = rf_model.predict_proba(X)[:, 1]
        xgb_pred = xgb_model.predict_proba(X)[:, 1]
        svm_pred = svm_model.decision_function(X)
        iso_pred = -iso_model.score_samples(X)

        # ✅ Autoencoder reconstruction
        scaled_data = scaler.transform(X)
        ae_recon = autoencoder_model.predict(scaled_data)
        ae_error = np.mean(np.square(scaled_data - ae_recon), axis=1)

        # ✅ Ensemble score
        ensemble_score = (rf_pred + xgb_pred + svm_pred + iso_pred + ae_error) / 5

        results = [
            {"Patient": int(i + 1), "Risk_Score": float(s), "High_Risk": bool(s > 0.5)}
            for i, s in enumerate(ensemble_score)
        ]

        print("✅ Prediction successful.")
        return {"status": "success", "results": results}

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
