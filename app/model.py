"""Wrapper around the trained multimodal deepfake detector (ResNet-50 + Wav2Vec 2.0).

The two functions marked TODO are the only places that depend on YOUR final-year
project code. Copy your model class and preprocessing into this project, then fill
them in. Until then, set DEMO_MODE=1 to run the whole API with a dummy detector.
"""
import os
import random
from pathlib import Path

import numpy as np

WEIGHTS_PATH = Path(os.getenv("WEIGHTS_PATH", "weights/deepguard.pt"))
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"


class DeepfakeDetector:
    def __init__(self):
        self.model = None
        self.device = None

    def load(self):
        if DEMO_MODE:
            return self
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO 1: import your model class and load the trained weights, e.g.
        #   from app.deepfake_model import FusionModel
        #   self.model = FusionModel()
        #   self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
        #   self.model.to(self.device).eval()
        raise NotImplementedError("Wire your trained model in app/model.py (TODO 1)")

    def predict(self, video_path: str) -> dict:
        """Return {"label": "real"|"fake", "confidence": float in [0, 1]}."""
        if DEMO_MODE:
            p_fake = random.random()
            return self._to_result(p_fake)

        import torch

        # TODO 2: reuse your preprocessing from the project:
        #   frames = extract_frames(video_path)      -> tensor for ResNet-50
        #   waveform = extract_audio(video_path)     -> tensor for Wav2Vec 2.0
        #   with torch.no_grad():
        #       logits = self.model(frames.to(self.device), waveform.to(self.device))
        #       p_fake = torch.softmax(logits, dim=-1)[0, 1].item()
        #   return self._to_result(p_fake)
        raise NotImplementedError("Add your preprocessing and forward pass (TODO 2)")

    @staticmethod
    def _to_result(p_fake: float) -> dict:
        p_fake = float(np.clip(p_fake, 0.0, 1.0))
        label = "fake" if p_fake >= 0.5 else "real"
        confidence = p_fake if label == "fake" else 1.0 - p_fake
        return {"label": label, "confidence": round(confidence, 4), "p_fake": round(p_fake, 4)}
