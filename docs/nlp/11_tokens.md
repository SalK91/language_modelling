# 11. Tokenization and Tokens
Tokenization defines the basic units over which a language model reasons and therefore strongly influences model capacity, efficiency, and generalization. Raw text is typically represented as Unicode strings, for example:

$$
\texttt{string = "Hello, How are you?"}
$$

Language models operate over sequences of tokens, which are usually represented as integer indices. For example:

$$
\texttt{indices = [15496, 11, 995, 0]}
$$

These integer indices serve as inputs to embedding layers, which map discrete tokens into continuous vector representations.

- A tokenizer is a class that implements two main functions:
  - encode: Converts a string into tokens (integers)
  - decode: Converts tokens back into a string
- The vocabulary size refers to the number of distinct tokens in the model's dictionary.

In practice, tokenizers are deterministic and shared between training and inference.

For interactive exploration, try:  [Link](https://tiktokenizer.vercel.app/?encoder=gpt2)

- Character-based tokenization: Character-based tokenizers split the input text into individual characters rather than words or subwords. This results in a much smaller and fixed vocabulary (e.g., all letters, digits, p. unctuation, and special characters). Every possible word can be represented, eliminating most out-of-vocabulary (OOV) issues

    - Pros:
        - Extremely small vocabulary size
        - Fully lossless representation — any text can be encoded and decoded perfectly
        - Robust to typos and unknown words — e.g., `"xylocopter"` can still be tokenized, even if it is a made-up word

    - Cons:
        - Characters are semantically weak — individual letters carry little meaning in Latin-based languages
        - Sequence lengths become much longer. For example, the word `"tokenization"` becomes 13 tokens instead of 1
        - More computationally expensive due to increased sequence length
        - May struggle to capture meaningful patterns unless large contexts are modeled

As a result, character-level models typically require deeper architectures or longer contexts to learn meaningful structure.

- Word-based tokenization:Splits text into words and punctuation using regular expressions.Each word is treated as a distinct token. For example, `"dog"` and `"dogs"` are different tokens even though they share semantic roots. Vocabulary size depends on the number of unique words in the corpus and is often extremely large. Word-based tokenization was the dominant approach in early statistical NLP systems.

    - Pros: Shorter sequences and semantically intuitive tokens
    - Cons:
        - Very large and variable vocabulary size
        - Morphologically similar words (e.g., `dog` vs `dogs`) are treated as unrelated tokens
        - Rare or unseen words require a special `[UNK]` (unknown) token
        - Vocabulary sparsity makes learning harder, especially for morphologically rich languages
These limitations are especially pronounced in languages with rich morphology or productive word formation.

## Subword Modeling

### Motivation and Linguistic Foundations
Subword modeling strikes a compromise between character-level and word-level tokenization.

Traditional NLP systems often assume a fixed-size vocabulary constructed from the training corpus, typically comprising tens or hundreds of thousands of words. This approach assigns a unique token such as `[UNK]` to all out-of-vocabulary (OOV) words encountered at inference time, introducing brittleness and data sparsity, especially for morphologically rich languages.

Many natural languages exhibit complex morphology, where individual words encode multiple grammatical or semantic features. For example, in Swahili, a single verb such as *ambia* (“to tell”) can yield hundreds of morphologically inflected forms, incorporating tense, aspect, negation, mood, definiteness, and object agreement. Relying on full-word tokenization under such conditions results in large vocabularies with sparse frequency distributions, undermining the statistical efficiency of learned models.

### Subword Units as a Solution

Subword modeling addresses the limitations of fixed-vocabulary word-based tokenization by decomposing words into smaller, more frequent units such as morphemes, syllables, or even individual characters. This approach enables:

- Generalization to unseen words via composition of known subword units
- Mitigation of the OOV problem by modeling open vocabularies
- Better cross-linguistic applicability, especially for agglutinative and polysynthetic languages
- More compact and data-efficient language representations

### Byte-Pair Encoding (BPE)

A prominent algorithm for data-driven subword segmentation is Byte-Pair Encoding (BPE), originally adapted from data compression and later introduced to NLP in neural machine translation. The core idea is to iteratively learn a vocabulary of subword units by merging the most frequent pairs of adjacent symbols in a corpus. BPE constructs subword units by greedily merging frequent symbol pairs.

The BPE algorithm operates as follows:

1. Initialize the vocabulary with all individual characters and a special end-of-word symbol (e.g., `_`)
2. Count all symbol pairs in the corpus and identify the most frequent pair
3. Merge the most frequent pair into a new symbol and update the corpus accordingly
4. Repeat steps 2–3 until the desired vocabulary size is reached

This procedure yields a deterministic and greedy segmentation strategy, allowing both training and inference to tokenize consistently using the learned merge rules. BPE produces variable-length subword units that often correspond to linguistically meaningful morphemes (e.g., prefixes, stems, suffixes), while maintaining robustness across diverse scripts and languages.

To illustrate the utility of BPE in handling both common and rare or noisy words, consider the following tokenization outcomes using a hypothetical learned subword vocabulary:

| Input Word | BPE Tokenization |
|-----------|------------------|
| `hat` | `hat` |
| `learn` | `learn` |
| `taaaaasty` | `taa## aaa## sty` |
| `laern` | `la## ern` |
| `Transformerify` | `Transformer##ify` |

The delimiter `##` indicates that a subword is not a standalone word but a continuation of a preceding segment. This example demonstrates several key advantages of subword modeling:

- Common words such as `hat` and `learn` are represented as single units
- Misspellings or informal variants (e.g., `laern`, `taaaaasty`) can still be tokenized into interpretable components using existing subword units
- Rare or novel words (e.g., `Transformerify`) are decomposed into known subwords, preserving semantic hints for downstream models

### Variants and Modern Usage

BPE has inspired several modern variants such as WordPiece and Unigram Language Model tokenization, which are widely used in state-of-the-art pretrained language models including BERT, GPT, and T5. These methods differ in how they select subword units, for example using likelihood-based criteria or probabilistic segmentation, but share the same underlying goal: modeling language compositionally below the word level.

Subword modeling constitutes a crucial innovation in modern NLP pipelines, striking a balance between granularity, efficiency, and linguistic fidelity.
