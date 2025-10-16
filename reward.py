from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentReward:
    """
    IMDB sentiment reward in [-1, 1]; +1 ~ positive, -1 ~ negative
    """

    def __init__(self, name: str, device: str):
        self.device = device
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(name).eval().to(device)
        )
        self.tok = AutoTokenizer.from_pretrained(name)

    @torch.no_grad()
    def score(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits  # (B, 2)
        probs = logits.softmax(-1)
        pos = probs[:, 1]
        return 2 * pos - 1.0
