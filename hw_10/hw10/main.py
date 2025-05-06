import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("lr_fitted.pkl", 'rb') as file:
    model = joblib.load(file)


class ModelRequestData(BaseModel):
    rooms: int
    floor: int
    lat: float
    lon: float


class Result(BaseModel):
    result: int


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.get("/predict_get/rooms_{rooms}_floor_{floor}_lon_{lon}_lat_{lat}")
def read_item(rooms: int, floor: int, lon: float, lat:float):
    input_df = pd.DataFrame(
        {
            "rooms": rooms,
            "floor": floor,
            "lat": lat,
            "lon": lon
        },
        index=[0],
    )
    result = model.predict(input_df)[0]
    return Result(result=int(result))

@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=int(result))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
