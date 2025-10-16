import torch
import torch.nn.functional as F
from utils import gather_logprob


def ppo_update(
    policy,
    optimizer,
    full_ids,
    gen_masks,
    old_logp,
    old_val,
    adv,
    ret,
    clip_range: float,
    clip_range_vf: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    ppo_epochs: int,
    mini_batch_size: int,
    pad_id: int,
):
    """
    One PPO inner loop over minibatches.
    Shapes:
      full_ids:   (B, T_total)
      gen_masks:  (B, T_total) in {0,1}, 1 only on generated region
      old_logp:   (B, T_total) (masked elsewhere)
      old_val:    (B, T_total)
      adv, ret:   (B, T_total)
    """
    device = full_ids.device
    B = full_ids.size(0)

    # Sequence-level shuffling
    order = torch.randperm(B, device=device)
    for _ in range(ppo_epochs):
        for start in range(0, B, mini_batch_size):
            mb = order[start : start + mini_batch_size]
            ids_mb = full_ids[mb]
            masks_mb = gen_masks[mb].float()
            adv_mb = adv[mb]
            ret_mb = ret[mb]
            old_lp_mb = old_logp[mb]
            val_old_mb = old_val[mb]

            mb_attn_full = (ids_mb != pad_id).long()
            out = policy.lm(
                ids_mb, attention_mask=mb_attn_full, output_hidden_states=True
            )
            logits = out.logits
            vals_new = policy.forward_value(out.hidden_states[-1])

            # token logprobs on chosen tokens
            logp_new = gather_logprob(logits, ids_mb) * masks_mb
            ratio = torch.exp(logp_new - old_lp_mb)  # token-wise

            # policy loss (masked mean)
            pg1 = ratio * adv_mb
            pg2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_mb
            denom = masks_mb.sum() + 1e-8
            policy_loss = -torch.sum(torch.min(pg1, pg2)) / denom

            # value loss (clipped)
            v_clipped = val_old_mb + torch.clamp(
                vals_new - val_old_mb, -clip_range_vf, clip_range_vf
            )
            v_loss1 = (vals_new - ret_mb) ** 2
            v_loss2 = (v_clipped - ret_mb) ** 2
            value_loss = 0.5 * torch.sum(torch.max(v_loss1, v_loss2) * masks_mb) / denom

            # entropy bonus
            ent = -(F.log_softmax(logits, -1) * F.softmax(logits, -1)).sum(-1)
            ent = (ent * masks_mb).sum() / denom

            loss = policy_loss + vf_coef * value_loss - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
