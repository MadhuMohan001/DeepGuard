import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.logger import log_prediction, summarize
from app.model import DeepfakeDetector

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = DeepfakeDetector().load()
    state["rag"] = None  # loaded lazily so the API starts fast
    yield


app = FastAPI(title="DeepGuard", version="1.0", lifespan=lifespan)


def get_rag():
    if state["rag"] is None:
        from app.rag import RagEngine

        state["rag"] = RagEngine()
    return state["rag"]


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": os.getenv("DEMO_MODE", "0") == "1"}


def _save_upload(video: UploadFile) -> str:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        return tmp.name


@app.post("/predict")
def predict(video: UploadFile = File(...)):
    path = _save_upload(video)
    try:
        result = state["detector"].predict(path)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    finally:
        Path(path).unlink(missing_ok=True)
    log_prediction(video.filename or "unknown", result)
    return result


@app.post("/predict_and_explain")
def predict_and_explain(video: UploadFile = File(...)):
    result = predict(video)
    explanation = get_rag().explain_prediction(result)
    return {"prediction": result, "explanation": explanation}


@app.post("/ask")
def ask(body: Question):
    return get_rag().answer(body.question)


@app.get("/stats")
def stats():
    return summarize()
