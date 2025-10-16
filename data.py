import random
from typing import List
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def load_imdb_texts(split: str = "train") -> List[str]:
    ds = load_dataset("imdb", split=split)
    return [x["text"] for x in ds]


def tokenize_batch(
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_len: int,
    device: str,
):
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def prompt_stream(
    texts: List[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    device: str,
    seed: int = 0,
):
    rng = random.Random(seed)
    while True:
        batch = rng.sample(texts, k=batch_size)
        yield tokenize_batch(tokenizer, batch, max_len, device)
