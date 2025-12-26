# Chapter 3: Window-based Neural Language Model

To address sparsity and storage issues, Bengio et al. proposed a Neural Probabilistic Language Model, the first large-scale deep learning model for NLP.

- Learns distributed representations (embeddings) for words  
- Uses a neural network to compute probabilities instead of raw counts  

Proposed in 2003, this model marked a shift from purely count-based language models to neural approaches that learn representations jointly with the language model.


### Neural Architecture (Simplified)

Given a fixed-length context window, the model estimates the conditional probability of the next word:
$$
P(w_t \mid w_{t-1}, \dots, w_{t-n})
$$


- Input: Concatenated word embeddings  
  $$
  \mathbf{e} = [e^{(1)}; e^{(2)}; e^{(3)}; e^{(4)}]
  $$

- Hidden layer:  
  $$
  \mathbf{h} = f(W \mathbf{e} + \mathbf{b}_1)
  $$

- Output (softmax over vocabulary):  
  $$
  \hat{y} = \text{softmax}(U \mathbf{h} + \mathbf{b}_2)
  $$

- Learns distributed representations (embeddings) for words that capture syntactic and semantic similarity.

### Full Model Equation

$$
\hat{y} = \text{softmax}\!\left(
W^{(2)} \tanh(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
+ W^{(3)} \mathbf{x} + \mathbf{b}^{(3)}
\right)
$$

- $W^{(1)}$ applied to word vectors (input $\rightarrow$ hidden layer)  
- $W^{(2)}$ applied to hidden layer (hidden $\rightarrow$ output)  
- $W^{(3)}$ connects input directly to output (shortcut connection)  

![Window-based Neural Language Model](images/fix_window_nn.png)

### Limitations of the Window-based Model

While the window-based neural language model marked a significant advancement, it still suffers from key limitations:

- Fixed context size (Markov assumption):  
  The model relies on a fixed-size context window (e.g., 4 or 5 previous words), enforcing a strong Markov assumption. This limits the model’s ability to capture long-range dependencies and contextual information beyond the window. As a result, important linguistic patterns that span wider contexts may be missed.

- Position-specific parameterization:  
  Each word in the context window is embedded and concatenated in a fixed order, meaning that words at different positions are treated differently by the model. This prevents the model from generalizing across positions; for example, the same word appearing in position 1 versus position 4 will be processed differently due to distinct weights applied in the hidden layer.

- Lack of permutation invariance and poor scalability:  
  Since the architecture depends on fixed positions and fully connected layers, it does not scale well with longer contexts and lacks flexibility for variable-length input. It also cannot handle unseen word orders or reorderings effectively.

These limitations were among the motivations for later architectures such as recurrent neural networks (RNNs) and transformers, which can model variable-length contexts and better capture sequential dependencies.
