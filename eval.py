import argparse, random, csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import EvalConfig
from reward import SentimentReward
from data import load_imdb_texts, tokenize_batch
from utils import set_seed, sample_generate, decode_gen_tail


@torch.no_grad()
def eval_batch(policy_lm, ref_lm, tok, rm, prompts, params: EvalConfig):
    # Tokenize prompts
    input_ids, attn = tokenize_batch(tok, prompts, params.max_prompt_len, params.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Generate with both models (same decoding settings)
    full_pol = sample_generate(
        policy_lm,
        input_ids,
        attn,
        params.gen_len,
        params.top_p,
        params.top_k,
        params.temperature,
        pad_id,
    )
    full_ref = sample_generate(
        ref_lm,
        input_ids,
        attn,
        params.gen_len,
        params.top_p,
        params.top_k,
        params.temperature,
        pad_id,
    )

    # Decode generated tails
    prompt_len = attn.sum(dim=1).long()  # (B,)
    texts_pol = decode_gen_tail(tok, full_pol, prompt_len, pad_id)
    texts_ref = decode_gen_tail(tok, full_ref, prompt_len, pad_id)

    # Sentiment extrinsic reward in [-1, 1]
    extr_pol = rm.score(texts_pol)  # (B,)
    extr_ref = rm.score(texts_ref)  # (B,)

    # For logging: generated length per sample
    B, T = full_pol.shape
    nonpad_len = (full_pol != pad_id).long().sum(dim=1)  # (B,)
    gen_len = (nonpad_len - prompt_len).clamp_min(0)  # (B,)

    return {
        "prompts": prompts,
        "texts_pol": texts_pol,
        "texts_ref": texts_ref,
        "extr_pol": extr_pol.cpu(),
        "extr_ref": extr_ref.cpu(),
        "gen_len": gen_len.cpu(),
    }


def main():
    params = EvalConfig()
    params.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(params.seed)

    # Tokenizer: left-padding + pad token for decoder-only models
    tok = AutoTokenizer.from_pretrained(params.ref_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    policy_lm = (
        AutoModelForCausalLM.from_pretrained(params.policy_path)
        .eval()
        .to(params.device)
    )
    ref_lm = (
        AutoModelForCausalLM.from_pretrained(params.ref_name).eval().to(params.device)
    )
    rm = SentimentReward(params.rm_name, params.device)

    # Held-out prompts
    test_texts = load_imdb_texts("test")
    rng = random.Random(params.seed)
    idx = list(range(len(test_texts)))
    rng.shuffle(idx)
    idx = idx[: params.n_samples]
    pool = [test_texts[i] for i in idx]

    rows = []
    wins_extr = 0
    total = 0

    for i in range(0, len(pool), params.batch_size):
        batch = pool[i : i + params.batch_size]
        out = eval_batch(policy_lm, ref_lm, tok, rm, batch, params)
        B = len(batch)
        total += B

        # Win count by extrinsic sentiment
        wins_extr += int((out["extr_pol"] > out["extr_ref"]).float().sum().item())

        for j in range(B):
            rows.append(
                {
                    "prompt": out["prompts"][j][:200].replace("\n", " ")
                    + ("..." if len(out["prompts"][j]) > 200 else ""),
                    "pol_text": out["texts_pol"][j][:200].replace("\n", " ")
                    + ("..." if len(out["texts_pol"][j]) > 200 else ""),
                    "ref_text": out["texts_ref"][j][:200].replace("\n", " ")
                    + ("..." if len(out["texts_ref"][j]) > 200 else ""),
                    "extr_pol": float(out["extr_pol"][j]),
                    "extr_ref": float(out["extr_ref"][j]),
                    "gen_len": float(out["gen_len"][j]),
                }
            )

    # Aggregates
    mean = lambda k: sum(r[k] for r in rows) / max(1, len(rows))
    pos_rate = lambda arr: sum(1.0 for x in arr if x > 0) / max(1, len(arr))

    extr_pol_list = [r["extr_pol"] for r in rows]
    extr_ref_list = [r["extr_ref"] for r in rows]

    print("\n=== Evaluation Summary (IMDB test) — Sentiment Only ===")
    print(f"Samples: {len(rows)} | gen_len_mean: {mean('gen_len'):.2f}")
    print(
        f"Extrinsic sentiment mean: policy={sum(extr_pol_list)/len(extr_pol_list):+.4f} | base={sum(extr_ref_list)/len(extr_ref_list):+.4f}"
    )
    print(
        f"% positive (policy) = {100.0 * pos_rate(extr_pol_list):.1f}% | % positive (base) = {100.0 * pos_rate(extr_ref_list):.1f}%"
    )
    print(f"Win-rate by sentiment (policy > base): {wins_extr/max(1,total):.3f}")

    # Save CSV
    if rows:
        with open(params.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-sample results to {params.out_csv}")


if __name__ == "__main__":
    main()
