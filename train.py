import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import TrainConfig
from model import GPT2WithValueHead
from reward import SentimentReward
from data import load_imdb_texts, prompt_stream
from utils import (
    set_seed,
    gather_logprob,
    gae,
    gen_mask_from_prompt,
    decode_gen_tail,
)
from ppo import ppo_update


def main():
    cfg = TrainConfig()
    # device handling
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # Tokenizer & models
    tok = AutoTokenizer.from_pretrained(cfg.policy_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # Important for decoder-only models (GPT-2)

    policy = GPT2WithValueHead(cfg.policy_name).to(cfg.device)
    ref = AutoModelForCausalLM.from_pretrained(cfg.policy_name).eval().to(cfg.device)

    rm = SentimentReward(cfg.rm_name, cfg.device)

    opt = torch.optim.AdamW(
        list(policy.lm.parameters()) + list(policy.value_head.parameters()),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    # Data
    texts = load_imdb_texts(split="train")
    stream = prompt_stream(
        texts, cfg.batch_size, tok, cfg.max_prompt_len, cfg.device, seed=cfg.seed
    )

    for step in range(1, cfg.outer_steps + 1):
        input_ids, attn = next(stream)

        with torch.no_grad():
            gen_ids = policy.generate(
                input_ids,
                attn,
                max_new_tokens=cfg.gen_len,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                temperature=cfg.temperature,
            )

        # Build generated-token mask
        full_ids = gen_ids
        gen_masks, prompt_len = gen_mask_from_prompt(attn, full_ids, tok.pad_token_id)

        # Attention mask for the full generated sequences
        attn_full = (full_ids != tok.pad_token_id).long()

        with torch.no_grad():
            out_pol = policy.lm(
                full_ids, attention_mask=attn_full, output_hidden_states=True
            )
            out_ref = ref(full_ids, attention_mask=attn_full)
        logp_pol = gather_logprob(out_pol.logits, full_ids)
        logp_ref = gather_logprob(out_ref.logits, full_ids)
        values = policy.forward_value(out_pol.hidden_states[-1]).detach()

        # Reward: terminal sentiment + KL penalty
        texts_tail = decode_gen_tail(tok, full_ids, prompt_len, tok.pad_token_id)
        term_reward = rm.score(texts_tail).to(cfg.device)  # (B,)

        kl = (logp_pol - logp_ref) * gen_masks
        rewards = -cfg.kl_coef * kl

        # add terminal sentiment reward at last generated token
        B = full_ids.size(0)
        for i in range(B):
            last_positions = torch.nonzero(gen_masks[i], as_tuple=False).squeeze(-1)
            if last_positions.numel() > 0:
                rewards[i, last_positions[-1]] += term_reward[i]

        # Mask out non-gen tokens for advantages
        rewards = rewards * gen_masks
        values = values * gen_masks

        adv, ret = gae(rewards, values, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_logp = logp_pol.detach() * gen_masks
        old_val = values.detach()

        # PPO updates
        ppo_update(
            policy,
            opt,
            full_ids,
            gen_masks,
            old_logp,
            old_val,
            adv,
            ret,
            clip_range=cfg.clip_range,
            clip_range_vf=cfg.clip_range_vf,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            max_grad_norm=cfg.max_grad_norm,
            ppo_epochs=cfg.ppo_epochs,
            mini_batch_size=cfg.mini_batch_size,
            pad_id=tok.pad_token_id,
        )

        if step % cfg.log_interval == 0:
            with torch.no_grad():
                avg_kl = (kl.sum() / (gen_masks.sum() + 1e-8)).item()
                print(
                    f"[Step {step}] mean sentiment={term_reward.mean().item():+.3f}  KL/token={avg_kl:+.3f}"
                )

    # Save checkpoint
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    policy.lm.save_pretrained(os.path.join(cfg.ckpt_dir, "policy_lm"))
    tok.save_pretrained(os.path.join(cfg.ckpt_dir, "policy_lm"))
    torch.save(
        policy.value_head.state_dict(), os.path.join(cfg.ckpt_dir, "value_head.pt")
    )
    print(f"Saved to {cfg.ckpt_dir}/")


if __name__ == "__main__":
    main()
