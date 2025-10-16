import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class GPT2WithValueHead(nn.Module):
    """
    GPT-2 policy LM + scalar value head on hidden states.
    """

    def __init__(self, name: str):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(name)
        hid = self.lm.config.n_embd
        self.value_head = nn.Linear(hid, 1)
        # ensure padding token exists
        self.lm.config.pad_token_id = self.lm.config.eos_token_id

    @torch.no_grad()
    def generate(
        self, input_ids, attention_mask, max_new_tokens, top_p, top_k, temperature
    ):
        return self.lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=self.lm.config.pad_token_id,
        )

    def forward_value(self, hidden_states):
        # hidden_states: (B, T, H)
        return self.value_head(hidden_states).squeeze(-1)  # (B, T)
