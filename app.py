import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", "rb") as f:
    saved_data = pickle.load(f)

vectorizer = saved_data["vectorizer"]
model = saved_data["model"]


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict_sentiment(user_input: TextInput):
    vectorized_text = vectorizer.transform([user_input.text])

    prediction = model.predict(vectorized_text)[0]

    return {"sentiment": prediction}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
