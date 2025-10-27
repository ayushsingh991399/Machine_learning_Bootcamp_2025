from fastapi import FastAPI
import pickle

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

@app.get("/")
def root():
    return {"message": "Model is ready"}

@app.post("/predict")
def predict(client: dict):
    X = [client]
    y_pred = model.predict_proba(X)[0, 1]
    return {"conversion_probability": y_pred}
