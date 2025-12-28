# Natural Language Generation

Natural Language Generation (NLG) is one of the two primary components of Natural Language Processing (NLP), the other being Natural Language Understanding (NLU). In simple terms:

$$
\text{NLP} = \text{NLU} + \text{NLG}
$$

While NLU focuses on interpreting and extracting meaning from human language, NLG is concerned with generating coherent, fluent, and contextually appropriate text intended for human readers.

Recent advances in deep learning have significantly improved the quality and capabilities of NLG systems, enabling them to produce natural-sounding language across a wide range of applications.

## Applications of Natural Language Generation

NLG plays a central role in many real-world applications, including:

- Machine translation (MT):
    - Input: Text in a source language
    - Output: Translated text in a target language

- Dialogue systems / digital assistants:
    - Input: Dialogue history or user queries
    - Output: Text that appropriately continues the conversation

- Text summarization:
    - Input: Long-form documents (e.g., research papers, emails, meeting transcripts)
    - Output: Concise and coherent summaries

- Creative writing and story generation

- Data-to-text generation:
    - Converts structured data (e.g., tables, databases) into natural language reports or descriptions

- Image and video captioning:
     - Generates natural language descriptions based on visual input

## Basics of Natural Language Generation

Most modern NLG systems rely on autoregressive language models, which generate text one token at a time. At each time step $t$, the model receives the preceding sequence of tokens $x_{<t}$ and predicts the next token $x_t$.

Let $f(\cdot)$ denote the model and $V$ be the vocabulary. For a given context $x_{<t}$, the model computes a score $s_v = f(x_{<t}, v) \in \mathbb{R}$ for each token $v \in V$. These scores are converted into probabilities via the softmax function:

$$
P(x_t = v \mid x_{<t})
=
\frac{\exp(s_v)}{\sum_{v' \in V} \exp(s_{v'})}
$$

### Architectures

- Non-open-ended tasks (e.g., machine translation):  
  Typically use an encoder-decoder architecture:
    - The encoder (often bidirectional) processes the input
    - The decoder (autoregressive) generates output one token at a time

- Open-ended tasks (e.g., story generation):  
  May rely solely on an autoregressive decoder without a separate encoder

### Training Objective

Models are trained using maximum likelihood estimation (MLE), which aims to maximize the probability of each token in the training sequence given its preceding context.

This is a token-level classification problem and is often trained using a method known as teacher forcing, where at each time step the model receives the true previous token instead of its own prediction. This aligns training conditions with the ground truth.
