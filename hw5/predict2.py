import pickle
from fastapi import FastAPI
from pydantic import BaseModel


# Define input schema
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


# Define response schema
class PredictResponse(BaseModel):
    conversion_probability: float
    will_convert: bool


# Initialize app
app = FastAPI(title="Conversion Prediction API")


# Load the pre-trained model from the base image
with open("/code/pipeline_v2.bin", "rb") as f_in:
    model = pickle.load(f_in)


@app.post("/predict", response_model=PredictResponse)
def predict(client: Client):
    client_dict = [client.dict()]
    prob = model.predict_proba(client_dict)[0, 1]
    will_convert = prob >= 0.5

    return PredictResponse(
        conversion_probability=float(prob),
        will_convert=will_convert
    )
