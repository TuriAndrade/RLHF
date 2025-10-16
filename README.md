# PPO Fine-Tuning of GPT-2 with Sentiment Reward (RLHF on IMDB)

This project implements **Reinforcement Learning from Human Feedback (RLHF)** using **Proximal Policy Optimization (PPO)** to fine-tune a small **GPT-2** model on the **IMDB movie reviews dataset**.  
The goal is to make the model generate **more positive** continuations when given a review prompt, while keeping it close to the base GPT-2 through a KL penalty.

---

## 🧠 Overview

- **Policy model**: GPT-2 (causal LM) fine-tuned with PPO  
- **Reward model**: `lvwerra/distilbert-imdb` (binary sentiment classifier)  
- **Reference model**: Frozen base GPT-2 (for KL regularization)  
- **Objective**

$$
\max_{\pi}\ \mathbb{E}_{\pi}\\left[
  r_T^{\text{sent}}
  \-\
  \beta \cdot D_{\mathrm{KL}}\big(
    \pi(\cdot \mid s_t)\ \|\ \pi_{\mathrm{ref}}(\cdot \mid s_t)
  \big)
\right]
$$

Here, $$\ r_T^{\text{sent}} \$$ is the terminal sentiment reward and $$\ \beta \$$ is the KL coefficient.


Each PPO update adjusts the GPT-2 weights to favor generations that the sentiment model judges as **more positive**, while preventing language drift via the KL term.

---

## 🏗️ Project Structure

```
.
├── train.py          # PPO training loop
├── eval.py           # Sentiment-only evaluation script
├── model.py          # GPT2WithValueHead (policy + critic)
├── reward.py         # DistilBERT-based sentiment reward model
├── ppo.py            # PPO inner-loop implementation
├── utils.py          # GAE, masking, decoding, generation helpers
├── data.py           # IMDB data loader and tokenizer
└── config.py         # Training and evaluation configs
```

---

## ⚙️ Training Procedure

1. Sample a batch of IMDB review prompts.
2. Generate continuations with the current GPT-2 policy.
3. Compute:
   - **Terminal sentiment reward** (from DistilBERT)
   - **KL penalty** (policy vs frozen base GPT-2)
4. Combine into **per-token rewards**

For generated tokens \( t = 1, \dots, T \):

$$
r_t =
\begin{cases}
-\beta \left(\log \pi(a_t \mid s_t) - \log \pi_{\mathrm{ref}}(a_t \mid s_t)\right), & t < T,\\
r_T^{\text{sent}}
\-\
\beta \left(\log \pi(a_T \mid s_T) - \log \pi_{\mathrm{ref}}(a_T \mid s_T)\right), & t = T.
\end{cases}
$$

6. Estimate **advantages and returns** using GAE(λ).
7. Update the policy and value networks via **clipped PPO**.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the policy
```bash
python train.py
```

This will:
- Fine-tune GPT-2 on IMDB via PPO,
- Log average sentiment and KL divergence per step,
- Save checkpoints to `ckpts/` (configurable via `cfg.ckpt_dir`).

### 3. Evaluate sentiment improvement
```bash
python eval.py
```

This compares the **fine-tuned** vs **base GPT-2** on the IMDB test split, reporting average sentiment and win-rate.

---

## 📊 Results

After training, evaluation on IMDB test prompts yielded:

| Metric | Policy (Fine-tuned) | Base GPT-2 |
|:--|:--:|:--:|
| **Extrinsic sentiment mean** | +0.4168 | +0.2732 |
| **% Positive generations** | 74.8 % | 65.6 % |

**Per-sample results** (prompts, generations, and scores) are saved in `eval_results.csv`.

The fine-tuned model consistently produces **more positive** continuations than the base GPT-2.
  
---

## 📘 References

- Christiano et al., *“Deep Reinforcement Learning from Human Preferences”*, NeurIPS 2017  
- Schulman et al., *“Proximal Policy Optimization Algorithms”*, arXiv 2017  
- Hugging Face Transformers: https://huggingface.co/docs/transformers  
