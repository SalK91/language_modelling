# Chapter 5: Long Short-term Memory
LSTMs were introduced specifically to address the vanishing gradient problem encountered in vanilla RNNs when modeling long sequences. LSTMs are a special kind of RNN capable of learning long-term dependencies. At each time-step $t$, the LSTM maintains a hidden state $h^{(t)}$ and a cell state $c^{(t)}$. The key components of an LSTM are the input, forget, and output gates, which regulate the flow of information.

![LSTM](images/lstm.png)

### Gates and Their Roles
Each gate is implemented as a sigmoid-activated affine transformation of the previous hidden state and the current input.

- Forget Gate: Controls what information from the previous cell state $c^{(t-1)}$ should be kept versus forgotten:
  $$
  f^{(t)} = \sigma(W_f h^{(t-1)} + U_f x^{(t)} + b_f)
  $$

- Input Gate: Determines what new information should be stored in the cell state:
  $$
  i^{(t)} = \sigma(W_i h^{(t-1)} + U_i x^{(t)} + b_i)
  $$

- Output Gate: Decides what part of the cell state should be output:
  $$
  o^{(t)} = \sigma(W_o h^{(t-1)} + U_o x^{(t)} + b_o)
  $$

### Internal Computation

- New cell content: this is the new content to be written to the cell:

    $$\tilde{c}^{(t)} = \tanh(W_c h^{(t-1)} + U_c x^{(t)} + b_c)$$

    
- Cell state update: erase (forget) some content from the last cell state and write (input) some new cell content:
  $$
  c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}
  $$

    The cell state therefore evolves through time via controlled addition and deletion of information, rather than complete overwriting.

- Hidden state output: read (output) some content from the cell:
  $$
  h^{(t)} = o^{(t)} \odot \tanh(c^{(t)})
  $$

    While the cell state stores long-term information, the hidden state serves as the short-term representation exposed to the rest of the network.

All gate outputs are vectors with values in $[0, 1]$ using the sigmoid function $\sigma$. The operator $\odot$ denotes element-wise (Hadamard) product.

LSTMs mitigate the vanishing gradient problem by enabling the cell state $c^{(t)}$ to carry forward important information over long sequences.

### LSTM: Additive Memory and Gradient Flow

One of the key strengths of LSTMs lies in the additive nature of its cell state update, which is visually emphasized in the diagram by the $\oplus$ (plus) operation in the center of the cell.

#### Why the “+” Is the Secret
The key architectural difference between LSTMs and vanilla RNNs lies in how information is propagated through time. Traditional RNNs use repeated multiplications when propagating hidden states across time, which can lead to gradients either vanishing (becoming very small) or exploding (becoming very large). This makes learning long-term dependencies extremely difficult.

LSTMs avoid this through the structure of their cell state update:
$$
c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}
$$

When the forget gate is close to 1 and the input gate is close to 0, the cell state behaves like an identity mapping across time-steps.

This equation is element-wise additive, rather than multiplicative. The additive interaction allows gradients to flow back through time more stably.

#### Mitigating Vanishing Gradients

- The forget gate $f^{(t)}$ controls how much of the previous cell state is retained. When $f^{(t)}$ is close to 1, $c^{(t-1)}$ is passed forward nearly unchanged.
- This allows information and gradients to flow across many time-steps with minimal decay.
- Because of the additive pathway through $c^{(t)}$, backpropagation through time (BPTT) can preserve useful gradients over long sequences.

#### Summary

The “$\oplus$” operator (additive memory update) is the core innovation in LSTMs. It allows the model to accumulate and preserve information over long time horizons, effectively addressing the vanishing gradient problem that affects vanilla RNNs.

![LSTM: Additive Memory](images/lstm_2.png)


This design makes LSTMs significantly more effective than vanilla RNNs for tasks involving long-range dependencies, such as language modeling and sequence labeling.