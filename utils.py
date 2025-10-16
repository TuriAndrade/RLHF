import random
import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def gather_logprob(logits, ids):
    # logits: (B,T,V), ids: (B,T) -> (B,T) chosen token logprobs
    logp = logits.log_softmax(-1)
    return logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def compute_kl(logp_pol, logp_ref):
    # token-wise KL estimate on sampled tokens (log pi - log pref)
    return logp_pol - logp_ref


def gae(rewards, values, gamma, lam):
    """
    rewards, values: (B, T) for generated region (masked elsewhere)
    Returns:
      adv, returns: (B, T)
    """
    B, T = rewards.size()
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_v = (
            values[:, t + 1] if t < T - 1 else torch.zeros(B, device=rewards.device)
        )
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[:, t] = lastgaelam
    returns = adv + values
    return adv, returns


def gen_mask_from_prompt(attn_prompt, full_ids, pad_id):
    """
    Build a (B,T_total) mask that's 1 on generated tokens and 0 elsewhere.

    attn_prompt: (B, T_prompt_padded) attention mask of the original prompt batch
    full_ids:    (B, T_total) token ids after generation (padded in batch)
    pad_id:      int
    """
    device = full_ids.device
    B, T = full_ids.size()

    prompt_len = attn_prompt.sum(dim=1).long()  # (B,)
    nonpad_len = (full_ids != pad_id).long().sum(dim=1)  # (B,)
    left_pad = T - nonpad_len  # (B,)

    start_gen = left_pad + prompt_len  # (B,)
    end_gen = left_pad + nonpad_len  # (B,)

    ar = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B,T)
    gen_masks = (ar >= start_gen.unsqueeze(1)) & (ar < end_gen.unsqueeze(1))
    return gen_masks.float(), prompt_len


def decode_gen_tail(tokenizer, full_ids, prompt_len, pad_id):
    """
    Decode only the generated part (after the prompt), robust to padding side.
    """
    texts = []
    B, T = full_ids.shape
    nonpad_len = (full_ids != pad_id).long().sum(dim=1)
    left_pad = T - nonpad_len
    start_gen = (left_pad + prompt_len.long()).tolist()
    for i in range(B):
        tail = full_ids[i, start_gen[i] :]
        texts.append(tokenizer.decode(tail, skip_special_tokens=True))
    return texts


@torch.no_grad()
def sample_generate(
    model, input_ids, attention_mask, max_new_tokens, top_p, top_k, temperature, pad_id
):
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=pad_id,
    )
