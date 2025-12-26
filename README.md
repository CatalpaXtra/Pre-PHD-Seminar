# Pre-PHD Seminar Syllabus
## Chapter Topic List
1. Gradient Descent & Backpropagation
2. Deep Neural Networks & Regularization
3. Transformers
4. LLMs
5. Multimodal Transformers
6. Reinforcement Learning
7. Mixture of Experts (MoE)
8. Diffusion Models
9. AI Infrastructure (GPUs & Parallelism)
10. Emerging AI architecture (Mamba, RetNet, RWKV, etc)

---

## Chapter 1: Gradient Descent & Backpropagation (Bishop DL Cp.7,8)
### Week 1 (Foundations)
- Reverse-mode autodiff vs. forward-mode; computational graphs; the chain rule in practice.
- From scalar backprop to vectorized backprop; Jacobian–vector products; gradient checking.
- Vanishing/exploding gradients, initialization & stabilization (layer norm preview).
- Derive backprop for a linear layer + nonlinearity; show how grad_fn chains compose.

### Week 2 (Engineering & Ideas)
- Dissect PyTorch’s autograd tape on toy MLP/CNN; write a custom autograd.Function (e.g., stable log-sum-exp).
- Implement a tiny autodiff engine to demystify backward(); extend to vector operations.
- Profiling and debugging gradients (anomaly mode, hooks, in-place gotchas).
- Mini-project: “Rebuild” a 2-layer net in ~100 lines that matches PyTorch’s gradients on CIFAR-10 subset.

### Additional Reading
- PyTorch Autograd mechanics & blog
- CS231n backprop notes
- Karpathy’s micrograd

---

## Chapter 2: Deep Neural Networks & Regularization (Bishop DL Cp.6,9)
### Week 1 (Foundations)
- BatchNorm; Mixup/CutMix.
- Bias–variance, L2/weight decay as constrained optimization; early stopping.
- Data augmentation.
- Dropout (intuition + model averaging view); label smoothing.
- Batch norm mechanics & pitfalls.
- Mixup/CutMix as vicinal risk minimization; when they help/hurt.

### Week 2 (Engineering & Ideas)
- Ablation lab: train a compact ResNet/MLP and toggle regularizers.
- Implement batch-norm from scratch; compare to PyTorch; study train/test stats drift.
- Augmentations: reproduce mixup/CutMix gains on CIFAR-10/100 with a fixed budget.

### Additional Reading
- Goodfellow Ch.7

---

## Chapter 3: Transformers (Bishop DL Cp.12.1-12.3)
### Week 1 (Foundations)
- Scaled dot-product attention; encoder vs. decoder; pre-LN vs. post-LN.
- Positional encodings (absolute, learned, RoPE).
- Depth/width/heads trade-offs; residual pathways & normalization for stability.
- Pre-LN/LayerNorm analysis; FlashAttention.

### Week 2 (Engineering & Ideas)
- Implement a minimal decoder block.
- Add RoPE; benchmark FlashAttention kernels vs. baseline.
- Long-context tricks: key-value caching, sliding-window attention.
- Reproduce a small Transformer on WikiText-2; report perplexity vs. context length.

### Additional Reading
- PyTorch blog on FlashAttention-3

---

## Chapter 4: LLMs
### Week 1 (Foundations)
- Why decoder-only for generative LLMs (causal LM & in-context learning); GPT-3 scaling.
- Alignment overview: supervised fine-tuning (SFT) vs. RLHF; InstructGPT methodology.
- Parameter-efficient tuning: LoRA core idea + variants (LoRA+).

### Week 2 (Engineering & Ideas)
- Build an SFT pipeline (Transformers Trainer) on a domain dataset.
- Fine-tune with LoRA (HF PEFT); measure memory/throughput vs. full-fine-tune.
- Prompting vs. SFT vs. LoRA ablation on the same tasks (e.g., summarization, QA).

### Additional Reading
- Build a LLM from scratch (Raschka)
- HuggingFace Transformers docs
- PEFT LoRA guide

---

## Chapter 5: Multimodal Transformers (Bishop DL Cp.12.4)
### Week 1 (Foundations)
- Vision Transformer (ViT): patchification, class token, pretrain-then-fine-tune; Swin’s shifted windows.
- Self-supervised ViTs: DINO; masked autoencoders (MAE).
- CLIP: contrastive image–text pretraining; zero-shot transfer.

### Week 2 (Engineering & Ideas)
- Train a tiny ViT or Swin on a small dataset; compare to CNN baseline.
- Zero-shot classification with CLIP; prompt/adapter tuning; retrieval demo.
- Mini-survey: “When does MAE beat supervised pretraining?” (reproduce a small ablation).

### Additional Reading
- Swin
- DINO
- MAE
- CLIP

---

## Chapter 6: Reinforcement Learning
### Week 1 (Foundations)
- On-policy policy-gradient recap.
- TRPO vs. PPO clips/KL penalties.
- PPO for language models (as in RLHF); monitoring KL, clip fraction, reward scaling.
- GRPO (Group Relative Policy Optimization): critic-free baselining from grouped scores; pros/cons.

### Week 2 (Engineering & Ideas)
- Classic RL lab: PPO on CartPole/LunarLander (Gymnasium or SB3); visualize learning dynamics.
- LLM-RL lab: TRL PPO or GRPO on a tiny text task (rule-based reward, e.g., length/regex or BLEU).
- Stress-test hyper-params (entropy bonus, KL-coef); “RL-for-LLMs” pitfalls.

### Additional Reading
- EasyRL
- PPO (Schulman et al.)
- GRPO (DeepSeekMath)
- Gymnasium

---

## Chapter 7: Mixture of Experts (MoE)
### Week 1 (Foundations)
- Conditional computation; top-k routing; load-balancing losses; expert capacity & token dropping.
- From Shazeer MoE to GShard to Switch Transformer; stability concerns & overflow.
- Compute vs. params: why sparse activation scales cheaply.

### Week 2 (Engineering & Ideas)
- Train a toy MoE layer; visualize gate distribution & token-to-expert traffic.
- Expert parallelism with DeepSpeed-MoE; compare throughput to dense baselines.
- Research sketch: dynamic routing with curriculum or domain-specialized experts.

### Additional Reading
- DeepSpeed MoE docs/tutorials

---

## Chapter 8: Diffusion Models (Bishop DL Cp.20)
### Week 1 (Foundations)
- DDPM forward/reverse processes; reweighting & variance schedules; likelihood vs. sample quality.
- DDIM accelerated sampling; classifier-free guidance; score-based SDE perspective.
- Latent diffusion (LDM/Stable Diffusion) and why latent spaces matter.

### Week 2 (Engineering & Ideas)
- Implement a minimal U-Net + DDPM on MNIST/CIFAR-10; add DDIM sampling head.
- Add classifier-free guidance, measure FID vs. sampling steps.
- Latent-diffusion lab: fine-tune a tiny LDM on a narrow concept; log speed/VRAM vs. pixel-space DDPM.

### Additional Reading
- DDPM
- DDIM
- Classifier-Free Guidance
- Score-SDE
- LDM

---

## Chapter 9: AI Infrastructure (GPUs & Parallelism)
### Week 1 (Foundations)
- GPU architecture essentials: tensor cores, memory hierarchy, mixed precision, NCCL collectives.
- Parallelism taxonomy: data, tensor (intra-layer), pipeline (GPipe), and ZeRO sharding; FSDP vs. DeepSpeed ZeRO.
- System-level throughput math: utilization, overlap (compute/comm), activation checkpointing.

### Week 2 (Engineering & Ideas)
- Scale a small Transformer with: (a) DDP baseline, (b) FSDP shard, (c) tensor + pipeline (Megatron/DeepSpeed); compare tokens/sec & cost.
- Memory-budgeting lab: fit the largest possible model on a single GPU via mixed precision + activation checkpointing; document trade-offs.
- Write a reproducibility one-pager: seeds, determinism, logging, experiment tracking.

### Additional Reading
- Programming Massively Parallel Processors (Wen-mei W. Hwu)
- NVIDIA A100 whitepaper
- Megatron-LM scaling case study
- PyTorch FSDP tutorials

---

## Chapter 10: Emerging AI architecture (Mamba, RetNet, RWKV, etc.)
### Week 1 (Foundations)

### Week 2 (Engineering & Ideas)

### Additional Reading

---