from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Features the model expects
REQUIRED_FEATURES = ['variance', 'skewness', 'curtosis', 'entropy']

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.post("/upload_csv_predict")
async def upload_csv_predict(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Check if all required features are present
        missing = list(set(REQUIRED_FEATURES) - set(df.columns))
        if missing:
            return {"error": f"Missing required columns in CSV: {missing}"}

        # Filter required columns
        filtered_df = df[REQUIRED_FEATURES]

        # Predict
        predictions = model.predict(filtered_df)

        return {
            "predictions": predictions.tolist()
        }

    except pd.errors.ParserError:
        return {"error": "Unable to parse CSV file. Please upload a valid CSV."}
    except Exception as e:
        return {"error": str(e)}
