## Chapter 4: Recurrent Neural Networks

Unlike traditional language models that condition on a fixed-size window of previous tokens, RNNs are capable of conditioning on the entire history of previous tokens.

This is achieved by maintaining a recurrent hidden state that summarizes past information and is updated at every time-step.

![A Recurrent Neural Network](images/rnn_3step.png)

## RNN Architecture

The RNN architecture (Figure above) shows three time-steps. Each hidden layer at time-step $t$ receives two inputs:

- The input vector at time $t$, $x_t$
- The hidden state from the previous time-step, $h_{t-1}$

These are combined using weights $W_{hh}$ and $W_{hx}$:

$$
h_t = \sigma(W_{hh} h_{t-1} + W_{hx} x_t + b_1)
$$

The hidden state $h_t$ acts as a continuous memory vector that compresses information from all previous inputs $x_1, \dots, x_t$.

The output is calculated as:

$$
\hat{y}_t = \text{softmax}(W_S h_t + b_2)
$$

Key parameters:

- $x_t \in \mathbb{R}^d$: Input word vector  
- $W_{hx} \in \mathbb{R}^{D_h \times d}$: Input weight matrix  
- $W_{hh} \in \mathbb{R}^{D_h \times D_h}$: Hidden state weight matrix  
- $W_S \in \mathbb{R}^{|V| \times D_h}$: Output weight matrix  
- $\sigma$: Non-linear function such as tanh  
- $\hat{y}_t \in \mathbb{R}^{|V|}$: Probability distribution over vocabulary  

## Training an RNN Language Model

To train a Recurrent Neural Network Language Model (RNN-LM), we begin with a large corpus of text, represented as a sequence of word tokens. The model is trained to predict the next word at each time-step, given all preceding words in the sequence.

At each time-step $t$:

- The model receives an input vector $x_t$ and computes a hidden state $h_t$ using the current input and the previous hidden state $h_{t-1}$.
- The output layer produces a probability distribution $\hat{y}_t$ over the vocabulary.
- The cross-entropy loss is computed between $\hat{y}_t$ and the ground-truth one-hot vector $y_t$.


Training an RNN language model corresponds to maximizing the log-likelihood of the observed sequence:

$$\log P(x_1, \ldots, x_T)
= \sum_{t=1}^{T} \log P\!\left(x_{t+1} \mid x_1, \ldots, x_t\right)$$


The total training objective is to minimize the average cross-entropy loss across all time-steps and sequences in the training set.

$$
J^{(t)}(\theta) = -\sum_{j=1}^{|V|} y_{t,j} \log \hat{y}_{t,j}
$$

### Teacher Forcing

During training, a technique called teacher forcing is commonly used. Instead of feeding the model’s own prediction $\hat{y}_{t-1}$ as input at the next time-step, we use the ground-truth word $y_{t-1}$. This stabilizes and accelerates learning by preventing the model from compounding its own mistakes during early training. However, it introduces a discrepancy between training and inference, known as exposure bias, since the model must rely on its own predictions at test time. Exposure bias becomes more pronounced for long sequences, where early prediction errors can cascade through the remainder of the sequence.

### Backpropagation Through Time

Training an RNN requires computing gradients over sequences of time-dependent operations. This is done using Backpropagation Through Time (BPTT):

- The RNN is unrolled over $T$ time-steps, creating a computational graph where the same parameters are shared at each step.
- Gradients are computed by propagating errors backward through time from $t = T$ to $t = 0$, summing contributions from all steps.
- This process allows the model to adjust its parameters based on dependencies that span multiple time-steps.

In practice, full BPTT over long sequences can be computationally expensive and unstable. Therefore, a variant called truncated BPTT is often used, where backpropagation is limited to a fixed number of time-steps (e.g., 20 or 50). This trades off full temporal context for efficiency and stability. Truncated BPTT implicitly limits the temporal horizon over which dependencies can be learned, acting as a practical approximation to full sequence optimization.

BPTT is sensitive to gradient vanishing and exploding problems, especially with non-linear activations like tanh or sigmoid. Techniques such as gradient clipping and using gated architectures like LSTMs or GRUs are commonly employed to address these issues.

### Efficient Training with Mini-Batches

Computing the loss and gradients over the entire corpus simultaneously is impractical due to memory limitations. Instead, training is performed on smaller units using mini-batches:

- The text corpus is divided into sequences such as sentences or fixed-length token chunks.
- A batch of these sequences is processed together to compute the average loss and gradients.
- Stochastic Gradient Descent (SGD) or its variants (e.g., Adam) are used to update parameters based on each batch.

Handling variable-length sequences within batches requires padding and masking:

- Shorter sequences are padded with special tokens to match the longest sequence in the batch.
- Padding positions are masked to prevent them from contributing to the loss or gradient updates.

This mini-batch training paradigm enables scalable computation and leverages hardware acceleration such as GPUs and TPUs.

### Training vs. Inference

At inference time, teacher forcing is not available. Each predicted word is fed back into the model as input for the next step. This difference in input distribution between training and inference can degrade performance if the model is not robust to its own prediction errors. Techniques such as scheduled sampling or fine-tuning with generated sequences are sometimes used to mitigate this mismatch.

### Perplexity

- The standard evaluation metric for language models is perplexity.

$$
\text{perplexity} =
\left(
\prod_{t=1}^{T}
\frac{1}{P_{\text{LM}}(x^{(t+1)} \mid x^{(1)}, \ldots, x^{(t)})}
\right)^{\frac{1}{T}}
$$

- Inverse probability of corpus according to the language model, normalized by the number of words.
- This is equal to the exponential of the cross-entropy loss $J(\theta)$:

$$
\left(
\prod_{t=1}^{T}
\frac{1}{\hat{y}^{(t)}_{x_{t+1}}}
\right)^{\frac{1}{T}}
=
\exp\left(
\frac{1}{T}
\sum_{t=1}^{T}
-\log \hat{y}^{(t)}_{x_{t+1}}
\right)
=
\exp(J(\theta))
$$

Perplexity can be viewed as the geometric mean of the inverse predicted probabilities assigned to the true next tokens.

Perplexity can be interpreted as the effective average number of choices the model is considering at each time-step:

- A perplexity of 1 means the model predicts every word perfectly.
- A perplexity of 50 means the model is as uncertain as choosing uniformly from 50 possible words.

Lower perplexity indicates better predictive performance and correlates with fluency and accuracy in language generation.

Note: Perplexity is sensitive to vocabulary size and tokenization. Models should be compared under identical preprocessing conditions.

## Memory Scaling in Recurrent Neural Networks

From a machine learning perspective, the memory requirements of RNNs can be divided into model parameters and runtime memory.

### Model Parameters (Static Memory)

These are the weights and biases defining the RNN structure:

- $W_{hx} \in \mathbb{R}^{D_h \times d}$
- $W_{hh} \in \mathbb{R}^{D_h \times D_h}$
- $W_S \in \mathbb{R}^{|V| \times D_h}$
- Bias vectors for each layer

The memory required to store these parameters is constant with respect to corpus size. That is, increasing the number of sentences or tokens in the dataset does not change the size of these matrices.

### Runtime Memory (Dynamic, per sequence)

During training with BPTT, the RNN stores:

- Input vectors $x_t$
- Hidden states $h_t$
- Intermediate gradient-related states

Runtime memory scales linearly with sequence length $T$. A sentence with $T$ words requires storage for $T$ hidden states and possibly $T$ sets of gradients.

- Model parameter count depends only on architecture parameters $D_h$, $d$, and $|V|$.
- Training memory grows proportionally with input sequence length.

Insight: RNNs are more memory-efficient than traditional $n$-gram models in terms of model size, but training on long sequences increases memory usage due to the need to store intermediate activations over time.

## Advantages and Disadvantages

Advantages:

1. Can handle variable-length sequences  
2. Parameter size independent of input length  
3. Can model long-range dependencies  
4. Weight sharing across time-steps  

Disadvantages:

1. Sequential computation limits parallelization  
2. Prone to vanishing and exploding gradients  
3. Difficulty in learning very long-term dependencies in practice

## Vanishing and Exploding Gradients

The total gradient with respect to parameters is:

$$
\frac{\partial E}{\partial W}
=
\sum_{t=1}^{T}
\sum_{k=1}^{t}
\frac{\partial E_t}{\partial y_t}
\cdot
\frac{\partial y_t}{\partial h_t}
\cdot
\left(
\prod_{j=k+1}^{t}
\frac{\partial h_j}{\partial h_{j-1}}
\right)
\cdot
\frac{\partial h_k}{\partial W}
$$

Jacobian norm bound:

$$
\left\|
\frac{\partial h_t}{\partial h_k}
\right\|
\le
(\beta_W \beta_h)^{t-k}
$$

Gradient clipping:

$$
\text{if } \|g\| \ge \text{threshold},
\quad
g \leftarrow
\frac{\text{threshold}}{\|g\|} \cdot g
$$

Solutions to vanishing gradients:

- Identity initialization for $W_{hh}$
- Use ReLU instead of sigmoid or tanh

While RNNs provide a principled framework for sequence modeling and variable-length context, their training difficulties and limited parallelism motivated the development of gated recurrent architectures and attention-based models, which we study next.