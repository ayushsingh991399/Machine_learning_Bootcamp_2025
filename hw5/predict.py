import pickle

with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = [client]
pred = model.predict_proba(X)[0, 1]
print(pred)
