from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib, json, numpy as np, os, requests
from tensorflow.keras.models import load_model

app = FastAPI(title="Health Risk Analyzer API", version="3.0")

# ✅ Model directory setup
model_dir = os.path.join(os.getcwd(), "mmodels")
os.makedirs(model_dir, exist_ok=True)

# ✅ Google Drive model links
model_urls = {
    "rf_model.pkl": "https://drive.google.com/uc?id=1vdYZCmWM1-IYhDqjt3LLfd25Xsl8eRHV",
    "xgb_model.pkl": "https://drive.google.com/uc?id=1SYhCbOkbR6qjHJ9kMYJwqeXh27lBnuFp",
    "svm_model.pkl": "https://drive.google.com/uc?id=15CgVxXcQzw_8any7CXjPzMSZzrC7RZNW",
    "iso_model.pkl": "https://drive.google.com/uc?id=1bOG11svcEN0FD7Bj-WuFr3HDmK9-tRFX",
    "autoencoder.h5": "https://drive.google.com/uc?id=1dArJ0cLPYP2DneEYdXZ6G5axb4XWFs7Q",
    "scaler.pkl": "https://drive.google.com/uc?id=1iY66eGcwVF-q6ogV91xBjB8Y9-yf-R4Y"
}

# ✅ Download models if missing
for fname, url in model_urls.items():
    path = os.path.join(model_dir, fname)
    if not os.path.exists(path):
        try:
            print(f"⬇️ Downloading {fname} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded {fname}")
        except Exception as e:
            print(f"⚠️ Failed to download {fname}: {e}")

# ✅ Safe model loader
def safe_load(path, loader, name):
    try:
        model = loader(path)
        print(f"✅ Loaded {name}")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load {name}: {e}")
        return None

rf_model = safe_load(f"{model_dir}/rf_model.pkl", joblib.load, "Random Forest")
xgb_model = safe_load(f"{model_dir}/xgb_model.pkl", joblib.load, "XGBoost")
svm_model = safe_load(f"{model_dir}/svm_model.pkl", joblib.load, "SVM")
iso_model = safe_load(f"{model_dir}/iso_model.pkl", joblib.load, "Isolation Forest")
autoencoder_model = safe_load(f"{model_dir}/autoencoder.h5", lambda p: load_model(p, compile=False), "Autoencoder")
scaler = safe_load(f"{model_dir}/scaler.pkl", joblib.load, "Scaler")

# ✅ Correct 25-feature input order from X_train_hybrid.csv
feature_order = [
    "age",
    "resting_blood_pressure",
    "cholestoral",
    "Max_heart_rate",
    "oldpeak",
    "target",
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

print(f"✅ Feature order confirmed: {len(feature_order)} features")

# 🩺 Root endpoint
@app.get("/")
def root():
    return {"message": "✅ Health Risk Analyzer API active and stable."}

# 🧠 Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        import io
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.strip() for c in df.columns]
        print(f"🧾 Uploaded CSV columns: {list(df.columns)}")

        if df.empty:
            return {"status": "error", "message": "Uploaded CSV is empty."}

        # ✅ Add missing columns
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_order].apply(pd.to_numeric, errors="coerce").fillna(0)
        X = df.to_numpy().astype(float)
        print(f"✅ Data ready for prediction: shape={X.shape}")

        preds = []

        # ✅ Random Forest
        if rf_model is not None:
            preds.append(rf_model.predict_proba(X)[:, 1])
        # ✅ XGBoost
        if xgb_model is not None:
            preds.append(xgb_model.predict_proba(X)[:, 1])
        # ✅ SVM
        if svm_model is not None:
            preds.append(svm_model.decision_function(X))
        # ✅ Isolation Forest
        if iso_model is not None:
            preds.append(-iso_model.score_samples(X))
        # ✅ Autoencoder
        if autoencoder_model is not None and scaler is not None:
            scaled = scaler.transform(X)
            recon = autoencoder_model.predict(scaled)
            ae_err = np.mean(np.square(scaled - recon), axis=1)
            preds.append(ae_err)

        if not preds:
            return {"status": "error", "message": "No models loaded successfully."}

        # ✅ Ensemble
        ensemble = np.mean(np.vstack(preds), axis=0)

        results = [
            {"Patient": int(i + 1), "Risk_Score": float(s), "High_Risk": bool(s > 0.5)}
            for i, s in enumerate(ensemble)
        ]

        print("✅ Prediction complete.")
        return {"status": "success", "results": results}

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
