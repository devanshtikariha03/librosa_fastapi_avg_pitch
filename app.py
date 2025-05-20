from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import requests, numpy as np, librosa
from io import BytesIO

app = FastAPI(title="Avg Pitch Service")

class PitchResponse(BaseModel):
    avgPitchHz: float

def fetch_audio(url: str):
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise HTTPException(400, "Fetch failed")
    return librosa.load(BytesIO(r.content), sr=None, mono=True)

def calc_pitch(y):
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    valid = f0[~np.isnan(f0)]
    return float(np.mean(valid)) if valid.size else 0.0

@app.get("/api/avg-pitch", response_model=PitchResponse)
async def avg_pitch(url: HttpUrl = Query(...)):
    y, _ = fetch_audio(str(url))
    return PitchResponse(avgPitchHz=calc_pitch(y))
