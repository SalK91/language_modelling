# Chapter 2: Language Models and Word Sequence Probability

Language models formalize the idea that natural language is not random, but follows statistical regularities. By assigning probabilities to word sequences, they provide a principled way to reason about which sentences are more likely than others.

### Purpose of Language Models

- A language model computes the probability of a sequence of words occurring.
- Useful in applications like machine translation and speech recognition.

At their core, language models answer a simple question: given what we have seen so far, what word is likely to come next?

### Probability of Word Sequences

- The probability of a word sequence $\{w_1, w_2, \ldots, w_m\}$ is:
  $$
  P(w_1, \ldots, w_m) = \prod_{i=1}^{m} P(w_i \mid w_1, \ldots, w_{i-1})
  $$
- Approximated using only the previous $n$ words (n-gram model):
  $$
  P(w_1, \ldots, w_m) \approx \prod_{i=1}^{m} P(w_i \mid w_{i-n}, \ldots, w_{i-1})
  $$

This approximation reflects a practical trade-off: modeling long-range dependencies is difficult with limited data and computation.

### Importance in Machine Translation

- Systems generate multiple candidate translations.
- Examples: {I have, I had, I has, me have, me had}
- Each candidate sequence is scored using a probability function.
- The system selects the one with the highest score.

In this setting, the language model acts as a fluency filter, favoring grammatically and statistically plausible sentences.

## N-gram Language Models

To compute word sequence probabilities, we can use the frequency of $n$-grams compared to the frequency of shorter sequences (e.g., uni-grams). This is the foundation of the n-gram language model.

These models rely purely on observed counts and make no use of semantic representations.

- Bigram model:
  $$
  P(w_2 \mid w_1) = \frac{\text{count}(w_1, w_2)}{\text{count}(w_1)}
  $$
- Trigram model:
  $$
  P(w_3 \mid w_1, w_2) = \frac{\text{count}(w_1, w_2, w_3)}{\text{count}(w_1, w_2)}
  $$

These models use a fixed-size window (e.g., previous $n$ words) to predict the next word. However, choosing the right window size is critical. For example:

- In the sentence: “As the proctor started the clock, the students opened their...”
- A 3-word context window (“the students opened their”) may predict “books.”
- A longer window capturing “proctor” may increase the likelihood of predicting “exam.”

This illustrates how limited context can restrict the model’s ability to capture meaning.

### Challenges with N-gram Models

#### 1. Sparsity

- If a specific $n$-gram (e.g., $(w_1, w_2, w_3)$) never appears, the model assigns a zero probability.
- Smoothing: Add a small value $\delta$ to all counts to avoid zero probabilities.
- Backoff: If the context (e.g., $w_1, w_2$) is missing, fall back to a shorter context (e.g., $w_2$ alone).
- Increasing $n$ makes sparsity worse, which is why typically $n \leq 5$.

Sparsity fundamentally limits how much context classical count-based models can exploit.

#### 2. Storage

- All $n$-gram counts must be stored.
- Larger $n$ and larger corpora increase model size significantly.

Together, sparsity and storage constraints motivate the transition from count-based language models to neural language models, which replace explicit counting with learned distributed representations.
