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

@app.post("/upload_csv_predict")
async def upload_csv_predict(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Filter only required columns, ignore extras
        filtered_df = df[REQUIRED_FEATURES]  # This will throw an error if any column is missing

        # Predict
        predictions = model.predict(filtered_df)

        # Return predictions with original index
        return {
            "predictions": predictions.tolist()
        }

    except KeyError as e:
        missing = list(set(REQUIRED_FEATURES) - set(df.columns))
        return {"error": f"Missing required columns in CSV: {missing}"}
    except Exception as e:
        return {"error": str(e)}
