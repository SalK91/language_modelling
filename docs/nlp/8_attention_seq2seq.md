# 8. Attention Mechanism in Sequence-to-Sequence Models


Traditional sequence-to-sequence (seq2seq) models suffer from a bottleneck where the entire source sentence is compressed into a single fixed-size vector (i.e., the final encoder hidden state). This poses challenges, particularly for long or information-rich inputs, as important details may be lost. This fixed-length bottleneck becomes increasingly problematic as input sequences grow longer or contain multiple salient elements.

The attention mechanism alleviates this issue by allowing the decoder to dynamically attend to different parts of the input sequence at each time step. Rather than depending solely on a single context vector, the decoder computes a weighted combination of all encoder hidden states, tailored for each output step. Intuitively, attention allows the decoder to query the encoder for the most relevant information at each decoding step.

![Attention - Seq-to-seq](images/attention_seq_t_seq.png)

### Mathematical Formulation

Assume the encoder outputs a sequence of hidden states. The encoder hidden states represent the source sequence, while the decoder hidden state represents the current generation context.

$$
h_1, h_2, \dots, h_N \in \mathbb{R}^h
$$

and the decoder has a hidden state at time step $t$:

$$
s_t \in \mathbb{R}^h
$$

The attention mechanism proceeds as follows:

1. Score computation: Compute unnormalized attention scores between the decoder hidden state and each encoder hidden state:

    $$
    e^t = [s_t^\top h_1, \dots, s_t^\top h_N] \in \mathbb{R}^N
    $$

    Each score measures the compatibility between the current decoder state and a specific encoder state.

2. Attention weights: Apply the softmax function to normalize these scores and obtain a probability distribution over the input positions:

    $$
    \alpha^t = \text{softmax}(e^t) \in \mathbb{R}^N
    $$

    The resulting weights form a probability distribution over input positions.

3. Context vector: Compute the attention output $a_t$ as the weighted sum of the encoder hidden states:

    $$
    a_t = \sum_{i=1}^N \alpha_i^t h_i \in \mathbb{R}^h
    $$

    This context vector changes at every decoding step, enabling dynamic focus over the input sequence.

4. Final decoder input: Concatenate the attention vector $a_t$ with the decoder hidden state $s_t$ to form a rich context vector for output generation:

    $$
    [a_t; s_t] \in \mathbb{R}^{2h}
    $$

This combined vector is typically passed through a feedforward layer or used directly for predicting the output token.

## Interpretability and Alignment

One of the key advantages of attention is that it provides a degree of interpretability to the model. Specifically:

- Alignment visualization: The attention weights $\alpha^t$ at each decoder step can be visualized to see which parts of the input the model is focusing on.
- Soft alignment: Attention naturally yields a soft alignment between source and target tokens without explicit supervision. The network learns this alignment purely from end-to-end training.
- Insight into model behavior: These alignments allow us to debug, explain, and understand the model's translation or generation decisions.

This emergent alignment capability is one of the reasons attention mechanisms are considered both powerful and elegant.

## Variants of Attention Mechanisms

While the basic attention mechanism uses a simple dot product between decoder and encoder hidden states, there are several alternative formulations that aim to improve the flexibility or expressiveness of the attention scoring function. Let $h_i \in \mathbb{R}^{d_1}$ denote the encoder hidden state at position $i$, and $s \in \mathbb{R}^{d_2}$ be the current decoder hidden state.

### 1. Dot-Product Attention (Luong et al.)

$$
e_i = s^\top h_i \in \mathbb{R}
$$

This is the simplest form of attention, assuming that $d_1 = d_2$. This method is efficient and often works well in practice, but lacks trainable parameters in the scoring function. Its simplicity makes it computationally efficient and well suited for large-scale models.

### 2. Multiplicative (Bilinear) Attention

$$
e_i = s^\top W h_i \in \mathbb{R}
$$

where $W \in \mathbb{R}^{d_2 \times d_1}$ is a learned weight matrix. This adds more expressiveness to the attention mechanism, enabling a trainable compatibility function between $s$ and $h_i$. 

### 3. Reduced-Rank Multiplicative Attention

$$
e_i = s^\top (U^\top V) h_i = (Us)^\top (Vh_i)
$$

with $U \in \mathbb{R}^{k \times d_2}$ and $V \in \mathbb{R}^{k \times d_1}$, where $k \ll d_1, d_2$. This low-rank factorization reduces the number of parameters and computations and is conceptually related to self-attention in Transformers. This formulation trades expressiveness for efficiency by constraining the attention interaction to a lower-dimensional subspace.

### 4. Additive (Bahdanau) Attention

$$
e_i = v^\top \tanh(W_1 h_i + W_2 s) \in \mathbb{R}
$$

where $W_1 \in \mathbb{R}^{d_3 \times d_1}$, $W_2 \in \mathbb{R}^{d_3 \times d_2}$ are learned weight matrices, and $v \in \mathbb{R}^{d_3}$ is a learned vector. Here, $d_3$ is a tunable dimensionality (sometimes called the attention dimension). Despite its name, this formulation involves a nonlinearity and trainable layers, making it functionally a feedforward neural network scoring mechanism. This form of attention was used in early neural machine translation systems and performs well when encoder and decoder dimensions differ.

Attention mechanisms eliminate the fixed-length context bottleneck and form the foundation for modern architectures such as Transformers, which rely entirely on attention for sequence modeling.