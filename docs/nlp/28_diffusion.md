# Diffusion in LLMs

Autoregressive LLMs are trained with teacher forcing using a causal mask, so training runs efficiently in parallel.  
Generation, however, is strictly sequential: each token must be produced one after another. This causes:

- latency proportional to sequence length $T$
- low accelerator utilization (small batch size during decoding)
- large memory traffic from KV-cache, which grows as $O(T)$
- inability to revise tokens globally once a prefix is committed

Diffusion models offer a different paradigm: instead of generating left→right, we generate by iteratively refining a fully corrupted sequence in parallel. This can reduce generation to $K$ refinement passes, where $K \ll T$, without relying on KV-cache.

## Diffusion: Image Generation Intuition (Primer)

In diffusion for images:

1. We define a forward process that gradually adds noise until the image becomes pure Gaussian noise.
2. Noise is easy to sample.
3. The model learns a reverse process that removes noise step by step to recover the image.

Forward corruption (simplified view):
$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$

Reverse model learns to predict and remove noise:
$
\mathcal{L} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t,t)\|^2
$

Modern samplers (e.g., DDIM, DPM-Solver) take larger, accurate denoising steps so that only $K$ passes are needed instead of $T$.


## How Diffusion is Adapted for Text and LLMs

Unlike images, text is discrete, so we cannot directly apply Gaussian noise to tokens. Two main adaptation paths exist:

### Discrete Token Diffusion

- The forward process corrupts tokens using masking or categorical replacement.
- The reverse process predicts all tokens in parallel and iteratively improves them.

A simple masking-based forward process:
$
x_t = \text{MASK}(x_0, \alpha_t)
$
where more tokens become MASK as $t$ increases.

Reverse model learns to fill masked positions using full context (bidirectional attention):
$
p_\theta(x_0^i | x_t, t) = \text{softmax}(h_\theta(x_t,t))_i
$

This family is often referred to as:

- Masked Diffusion Model (MDM) when masking is used as the corruption process
- MaskGIT-style decoding when inference fills tokens by confidence ordering in $K$ rounds

### Continuous (Embedding/Latent) Diffusion for Text

- Tokens are mapped to embeddings
- Embeddings are noised with Gaussian diffusion (like images)
- Denoiser reconstructs embeddings
- Final step projects to nearest tokens or softmax distribution

This is used in models like Diffusion-LM, enabling smooth controllability and higher-order samplers.


## Why Diffusion Helps LLM Decoding

Diffusion-based LLM decoding has different properties from AR:

| model | training | inference | memory |
|---|---|---|---|
| AR | parallel via causal mask | sequential $T$ passes | KV-cache grows as $O(T)$ |
| Diffusion | parallel denoising training | $K$ parallel refinement passes | constant memory, no KV-cache |

Key advantages:
- Token-parallel inference → full GPU/TPU compute used every pass
- Fewer total passes $K \ll T$ → lower latency
- No KV-cache growth → lower memory bandwidth pressure
- Whole-sequence refinement → globally coherent corrections each pass
- Supports controllable generation when done in embedding space


## Practical Implementation Notes (Non-Math Focus)

For applying diffusion to LLMs at scale:

1. Denoiser architecture  
   Usually still a transformer, but:
   
   - uses bidirectional attention, not causal
   - takes timestep or noise-level encoding as input
   - predicts token logits or clean embeddings, depending on design

2. Noise schedule  
   Instead of thinking in exact distributions, we define a simple increasing corruption schedule — early steps slightly corrupt, later steps heavily corrupt.

3. Fast sampling strategy
   - If operating in embedding space, we can use DDIM or DPM-Solver to jump across timesteps in 10–20 steps.
   - If operating in token space, we use iterative parallel masked prediction, often with confidence-based ordering, typically 5–15 passes.

4. Decoding mechanics
   - Each pass predicts all token positions at once
   - We sample or choose top tokens for corrupted positions
   - Lower confidence positions may be masked again for the next pass


## Limitations and Open Challenges

- Discrete diffusion requires careful corruption design for large vocabularies
- Embedding diffusion can suffer from embedding↔token mismatch at decoding time
- Quality often drops if $K$ is pushed too low
- Guidance and control mechanisms for discrete diffusion are still evolving
- Likelihood evaluation is harder than AR, so evaluation focuses more on generation quality and speed trade-offs
