from fastapi import FastAPI
import pandas as pd
from train_eval import train_and_evaluate

app = FastAPI(title="Sampling Assignment API")

@app.get("/run")
def run_pipeline():
    df = pd.read_csv("../data/Creditcard_data.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    results = train_and_evaluate(X, y)
    return results
