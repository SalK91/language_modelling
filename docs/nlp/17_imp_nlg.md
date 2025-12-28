
# Improving Training and Generation for NLG


## Improving Decoding in Language Generation

Decoding strategies play a crucial role in transforming a model’s token probabilities into coherent and high-quality output. Below are commonly used techniques and strategies to improve decoding.

### Greedy Decoding

- At each time step, select the token with the highest probability:
$$
\hat{y}_t = \arg\max_{w \in V} P(w \mid y_{<t})
$$
- Simple and fast, but may yield repetitive or suboptimal text due to lack of exploration.

### Beam Search

- Keeps the top-$k$ most likely partial sequences (beams) at each time step.
- Aims to maximize the overall sequence log-probability:
$$
\text{Score}(\hat{y}_{1:t}) = \sum_{i=1}^{t} \log P(\hat{y}_i \mid \hat{y}_{<i})
$$
- Length normalization is often applied to prevent a bias toward shorter sequences:
$$
\text{Score}_{\text{norm}}(\hat{y}_{1:t}) =
\frac{1}{t^\alpha} \sum_{i=1}^{t} \log P(\hat{y}_i \mid \hat{y}_{<i}),
\quad \alpha \in [0, 1]
$$

### Sampling-Based Methods

#### Top-$k$ Sampling

- At each time step, restrict sampling to the top-$k$ tokens in the probability distribution.
- Higher $k$ increases output diversity but may introduce incoherence.
- Lower $k$ leads to safer, more predictable outputs.

#### Top-$p$ (Nucleus) Sampling

- Dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$:
$$
\sum_{w \in V_p} P(w \mid y_{<t}) \geq p
$$
- More adaptive than top-$k$, especially when token probability mass is unevenly distributed.

### Temperature Scaling

- Temperature controls the sharpness of the softmax distribution:
$$
P_{\tau}(y_t = w)
=
\frac{\exp(s_w / \tau)}{\sum_{w' \in V} \exp(s_{w'} / \tau)}
$$
- $\tau > 1$: flattens the distribution, increasing randomness and diversity.
- $\tau < 1$: sharpens the distribution, making outputs more deterministic.

### Re-ranking Generated Sequences

- Decode multiple candidate sequences (e.g., 10), then rank them by a quality metric.
- Perplexity is a common metric but often favors generic or repetitive text due to high token likelihoods.
- Advanced re-rankers may consider:
    - Style
    - Discourse coherence
    - Factuality and entailment
    - Logical consistency
- Multiple re-rankers may be combined, but calibration issues can arise.



## Improving Training for Natural Language Generation Models

Training natural language generation (NLG) models with maximum likelihood estimation (MLE) using teacher forcing is effective but suffers from critical limitations. This section outlines the central issues, mitigation techniques, and advanced strategies used in both academic and production systems.

### Exposure Bias

- During training, the model conditions on ground-truth tokens $y_{<t}^*$; at test time, it conditions on its own previous predictions $\hat{y}_{<t}$.
- This mismatch can cause errors to compound—early mistakes can derail generation.

The standard teacher forcing objective is:
$$
\mathcal{L}_{\text{MLE}} =
- \sum_{t=1}^{T} \log P_\theta(y_t^* \mid y_{<t}^*)
$$

### Scheduled Sampling

- With probability $p$, the model uses its own previous prediction instead of the ground-truth token.
- $p$ increases over training to gradually expose the model to its own distribution.
- However, this violates the assumption that training inputs are independent of predictions, leading to inconsistent gradients and potential instability

### Dataset Aggregation

- Periodically generate sequences using the current model.
- Add these self-generated sequences to the training set.
- The model learns to correct its own outputs over time.

### Retrieval-Augmented Generation

- Retrieve prototype sentences from a corpus.
- Two common paradigms:
    - Edit-based generation: learn to modify retrieved examples.
    - Search-and-generate: use retrieval as a prompt (e.g., KNN-LM, RAG, RETRO).
- Grounding generation in human-written text improves fluency and informativeness.

### Reinforcement Learning for Text Generation

- Model generation as a Markov Decision Process:
    - States: current context representation
    - Actions: next-token choices
    - Policy: decoder distribution
    - Rewards: external scalar signal (e.g., BLEU)
- Objective: maximize expected total reward:
$$
\mathbb{E}_{\pi_\theta} \left[ \sum_{t=1}^{T} r_t \right]
$$
- Methods include REINFORCE, actor-critic, and policy gradients.

#### Limitations of Reinforcement Learning

- Sample inefficiency and high gradient variance.
- Difficult credit assignment over long sequences.
- Risk of overfitting the reward metric rather than true quality.

### Defining Reward Functions

Common automatic metrics include:
- BLEU for machine translation
- ROUGE for summarization
- CIDEr and SPIDEr for image captioning

These metrics are proxies and may correlate poorly with human judgment.

### Rewarding Specific Behaviors

Reinforcement learning can target fine-grained objectives:
- Politeness
- Simplicity
- Temporal coherence
- Cross-modality alignment
- Formality
- Human preference via reinforcement learning from human feedback (RLHF)

### Training Takeaways

- Teacher forcing is standard but introduces exposure bias.
- Mitigation strategies include scheduled sampling, dataset aggregation, retrieval augmentation, and reinforcement learning.
- Reinforcement learning is powerful but expensive and should be used when desired behaviors are not captured by likelihood-based training.
- Human feedback and reward modeling are central to aligning modern language models with user preferences.
