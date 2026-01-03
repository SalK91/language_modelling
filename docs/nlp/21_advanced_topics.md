# Advanced Topics in Language Modelling


## Knowledge Distillation for Language Models

Knowledge distillation is a model compression and capability transfer technique in NLP and LLMs that trains a compact student model to approximate a high-capacity teacher model. Instead of learning only from one-hot labels, the student also learns from the teacher’s soft predictive distribution and, optionally, its intermediate reasoning samples, improving both generalization and reliability while reducing inference cost.

### Teacher and Student Distributions

Given input $x$, the teacher outputs logits $z_t \in \mathbb{R}^{|V|}$ over vocabulary $V$. A temperature-scaled softmax produces softened probabilities:

$$
p_t^T = softmax(z_t / T), \quad T > 1
$$

The student model generates its own logits $z_s$ and corresponding distribution:

$$
p_s^T = softmax(z_s / T)
$$

When $T > 1$, distributions become smoother, exposing relationships between alternative token choices. This is critical for multi-step reasoning, where each token builds on implicit intermediate deductions. Distillation transfers not only correct answers but also richer decision structure and uncertainty.

### Distillation Loss

Training minimizes a weighted objective balancing:

- Correctness using cross-entropy with ground truth $y$
- Imitation using KL divergence between teacher and student distributions

Loss formulation:

$$
L = \alpha T^2 \, KL(p_t^T \| p_s^T) + (1 - \alpha) \, CE(y, softmax(z_s))
$$

Where:

$$
KL(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

The term $T^2$ rescales gradients to preserve signal magnitude when using high temperature, preventing vanishing updates.

 
### Representation Alignment (Encoder Models)

For encoder models, teacher and student hidden states $h_t$ and $h_s$ may differ in size. A learned projector $W$ aligns them:

$$
L_{rep} = MSE(h_t W, h_s)
$$

Alternative alignment objectives include cosine similarity or attention map matching across selected layers $\mathcal{L}_k$.

 
### Distillation Algorithm

#### Initial Setup

- Teacher Model: Large LM (e.g., 3.7B–175B parameters)
- Student Model: Smaller LM (e.g., 125M–1.3B parameters)
- Training Inputs: $\mathcal{D}_{Train} = \{x_i\}$
- Prompt Set: $P = \{(x_i, y_i, z_i)\}$ where $z_i$ are teacher reasoning samples

#### Sampling Process

For each example $x_i$:

1. Sample $N$ teacher reasoning–prediction pairs:
   $$
   (\hat{y}_i, \hat{z}_i) \sim \mathcal{N}_T(y_i, z_i \mid x_i, P)
   $$
2. Construct sample set:
   $$
   P_i = \{(x_i, \hat{z}_i^j, \hat{y}_i^j)\}_{j=1}^N
   $$
3. Typical setting:
   $$
   N = 30
   $$

Create corpus:

$$
C = \{(x_i, (\hat{z}_i^j, \hat{y}_i^j))\}_{j=1}^N
$$

 
#### Training Process

Train student on teacher samples using LM objective:

$$
L(z_s, y_s \mid C) = CE(\hat{y}_i^j, \hat{z}_i^j \mid x_i)
$$

This objective integrates into the full distillation loss defined earlier.

 
#### Evaluation Options

After training, evaluate output quality using:

- Greedy Decoding
  $$
  (\hat{z}_{test}, \hat{y}_{test}) = argmax_{z_s} \mathcal{S}(y \mid z_t)
  $$

- Self-Consistency
  $$
  \hat{y}_{test} = argmax_y \, \mathbb{E}_{z_s \sim \mathcal{S}_{top-k}} [\mathcal{S}(y \mid z_s, z_{test})]
  $$

 
#### Optional Generative Extension: Sample-and-Rank

1. Sample $n$ teacher completions per prompt
2. Score using reward model or teacher log-likelihood
3. Select top-$k$ sequences $\mathcal{S}_{top}$
4. Train student via:
   - $CE(\mathcal{S}_{top}, p_{student})$, or
   - policy distillation on teacher action distributions

---

## Mixture of Experts

In LLMs and NLP, mixture of experts (MoE) replaces dense feed-forward layers with a set of expert networks $\{E_1, ..., E_N\}$ and a router $G$ that selects a sparse subset of experts per token. Given input representation $h$, the router outputs gating logits $r = G(h)$ and expert selection is typically top-$k$:

$$
g_i = softmax(r)_i,\quad \mathcal{I} = topk(r, k)
$$

Only experts in $\mathcal{I}$ are executed, producing $e_i = E_i(h)$, and the MoE layer output is:

$$
y = \sum_{i \in \mathcal{I}} g_i \, e_i
$$

Training minimizes the task loss $L_{task}$ (e.g., next-token cross-entropy) plus auxiliary load-balancing and routing regularization to prevent expert collapse and imbalance. A common load-balancing loss uses batch-level expert utilization $f_i$ and mean gate probability $\bar{g}_i$:

$$
L = L_{task} + \lambda \sum_{i=1}^N f_i \cdot \bar{g}_i
$$

Alternative formulations use entropy bonuses on $g$, z-loss on router logits, or differentiable routing approximations.

Algorithm steps:

1. Replace MLP layers with MoE block containing $N$ experts and router $G$.
2. For each token, compute router logits $r = G(h)$.
3. Select expert indices $\mathcal{I} = topk(r, k)$.
4. Compute gates $g_i = softmax(r)_i$ for $i \in \mathcal{I}$.
5. Execute selected experts $e_i = E_i(h)$.
6. Aggregate $y = \sum_{i \in \mathcal{I}} g_i \, e_i$.
7. Compute loss $L = L_{task} + L_{aux}$ and backprop to student and router.
8. Optionally apply capacity limits per expert, expert dropout, router jitter noise, or switch to single-expert routing (top-1) variants.

Efficiency note: MoE increases parameter count while reducing per-token FLOPs via sparse routing, improving scaling at controlled inference cost.

---

## Predicting the Next Token in Language Models

1. Greedy Decoding: Greedy decoding selects the token with the highest predicted probability at each step: 

    $$x_{t+1} = \arg\max_{x} \, p(x \mid x_{1:t})$$  
    
    While simple and efficient, it often produces suboptimal sequences that are repetitive, lack diversity, or fail to capture long-range coherence, since it ignores alternative plausible continuations.

2. Beam Search: Beam search maintains the top-$k$ most probable partial sequences (beams) at each timestep. The algorithm expands each beam with all possible next tokens, retains the $k$ highest-scoring sequences, and repeats until termination. Benefits include higher likelihood sequences compared to greedy decoding, but beam search increases computational cost and can still lack diversity, often producing deterministic or generic outputs if the beam width is small or length penalties are not applied.

3. Sampling-Based Decoding: Sampling introduces stochasticity to token selection, enabling more diverse and natural outputs. Common strategies include:  

    - Top-$k$ sampling: Restrict the candidate set to the $k$ most probable tokens and sample according to their normalized probabilities.  

        $$p'(x \mid x_{1:t}) = \frac{p(x \mid x_{1:t})}{\sum_{i \in topk} p(i \mid x_{1:t})}, \quad x \sim p'$$

    - Top-$p$ (nucleus) sampling: Select the smallest set of tokens whose cumulative probability exceeds a threshold $p$ and sample from this set.  
        
        $$\mathcal{V}_p = \min \{V' \mid \sum_{i \in V'} p(i \mid x_{1:t}) \ge p \}, \quad x \sim p(i \mid i \in \mathcal{V}_p)$$

    - Temperature scaling: Adjusts the sharpness of the predicted distribution to control randomness:  
     
    $$p_T(x \mid x_{1:t}) = softmax \left( \frac{\log p(x \mid x_{1:t})}{T} \right)$$
    
    Low temperatures ($T<1$) make the distribution peakier, favoring high-probability tokens and reducing diversity. High temperatures ($T>1$) flatten the distribution, increasing randomness and creative outputs.

Overall, the choice of decoding strategy involves a tradeoff between likelihood, diversity, and computational efficiency. Greedy and beam search optimize for probability, whereas sampling-based methods enhance diversity and naturalness in generated text.


---
## Inference Optimization in LLMs

### KV caching  
In autoregressive transformers, each new token attends to all prior tokens. Instead of recomputing past key/value projections, we cache them. For token step $t$, the attention computation becomes:

$$
Q_t = W_Q h_t,\quad K_{1:t} = [K_{cache}; W_K h_t],\quad V_{1:t} = [V_{cache}; W_V h_t]
$$

$$
A_t = softmax\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d_k}}\right) V_{1:t}
$$

The cache stores $(K_{1:t-1}, V_{1:t-1})$ from earlier steps. This reduces per-token FLOPs from $O(t)$ to $O(1)$ for projections, leaving only the attention dot product with cached states. Latency improves significantly for long contexts, at the cost of $O(n_{layers} · seq_{len} · d_{kv})$ memory.

  Algorithm steps:

  1. For each layer, compute $K = W_K h, V = W_V h$ for prompt tokens.
  2. Store $(K, V)$ in cache.
  3. During generation, compute only $K_t, V_t$ for the new token.
  4. Append to cache and attend using the full cached $K, V$.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)  

MQA shares a single key/value head across all query heads, reducing memory and bandwidth:

   $$
   K = W_K h,\quad V = W_V h \quad \text{(1 shared head)}
   $$

   All query heads attend to the same $K, V$. This cuts cache size by $h$× (number of Q heads).  
   GQA generalizes this by sharing $K, V$ across groups of query heads:

   $$
   \mathcal{G}_j = \{Q_{j,1},...,Q_{j,m}\},\quad K_j, V_j \text{ shared per group}
   $$

   If there are $h_q$ query heads and $h_{kv}$ KV heads, each KV head serves $h_q / h_{kv}$ query heads. GQA balances quality and efficiency better than full sharing (MQA) while still reducing memory bandwidth and cache footprint.


   Benefits:

   - Smaller KV cache ($↓$ memory, $↓$ GPU bandwidth pressure)
   - Faster decoding, especially in memory-bound regimes
   - Better quality than MQA when $h_{kv} > 1$

   Limitations:

   - Some capacity loss from reduced key/value specialization
   - Requires careful grouping choice to avoid quality drop

### PagedAttention  
Standard KV caches allocate contiguous memory per sequence, causing fragmentation and over-allocation when sequences vary in length. PagedAttention stores KV blocks in fixed-size pages (like virtual memory), allowing non-contiguous storage:

   Idea:

   - Preallocate memory into pages of size $P$.
   - Store $(K, V)$ in page slots, not one long tensor.
   - Map logical token positions → physical page addresses.

   Memory model:

   $$
   (K, V)_{layer} \in \mathbb{R}^{n_{pages} × P × d_{kv}}
   $$

   Attention reads KV by gathering relevant pages:

   $$
   A_t = softmax\left(\frac{Q_t K_{pages}^\top}{\sqrt{d_k}}\right) V_{pages}
   $$

Benefits:

   - Eliminates wasted padding memory
   - Enables large-batch serving without OOM
   - Efficient for dynamic and very long contexts
   - Reduces fragmentation and improves throughput

Algorithm steps:

   1. Partition prompt tokens into pages of length $P$.
   2. Store each page’s $K, V$ into free page slots.
   3. Maintain a page table mapping token index → page slot.
   4. During decoding, gather only required pages for attention.
   5. Append new token KV into next free page.

 

