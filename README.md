# DeepGuard: Deepfake Detection API with RAG Explanations

FastAPI service that serves a multimodal (ResNet-50 + Wav2Vec 2.0, cross-attention) PyTorch
deepfake detector, and a LangChain + ChromaDB RAG assistant that explains predictions and
answers questions using only the project's own paper and notes.

## Setup
```bash
pip install -r requirements.txt
cp your_paper.pdf docs/            # your IEEE paper / report / notes
python ingest.py                   # builds the ChromaDB index
export ANTHROPIC_API_KEY=...       # optional; without it /ask returns retrieved context only
uvicorn app.main:app --reload
```
Docs UI: http://127.0.0.1:8000/docs

## Wire in your trained model
Edit the two TODOs in `app/model.py`, then put weights at `weights/deepguard.pt`.
To try the API without the model: `DEMO_MODE=1 uvicorn app.main:app --reload`.

## Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /predict` | upload a video, get label + confidence |
| `POST /predict_and_explain` | prediction plus RAG explanation |
| `POST /ask` | ask a question about the project (RAG) |
| `GET /stats` | Pandas/NumPy summary of logged predictions |
| `GET /health` | status |

## Stack
Python, PyTorch, FastAPI, LangChain, ChromaDB, sentence-transformers, Pandas, NumPy.
