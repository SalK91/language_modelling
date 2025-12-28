# 13. The Pretraining Strategies
This section describes how pretraining objectives differ across encoder, decoder, and encoder–decoder architectures, and why these differences matter.

The pretraining/fine-tuning paradigm has become the dominant approach in modern NLP. Instead of training models from scratch for every task, we first train a general-purpose language model on a large, unlabeled corpus, and then adapt it to a specific task using a smaller labeled dataset. This approach enables models to transfer general linguistic knowledge learned during pretraining into a wide variety of downstream applications.

- Step 1: Pretraining — Train on a large-scale unsupervised objective such as language modeling. The model learns broad statistical regularities, syntax, semantics, and even some world knowledge from raw text
- Step 2: Fine-tuning — Adapt the pretrained model to a specific supervised task (e.g., sentiment classification, question answering, named entity recognition) using a smaller, labeled dataset



![Pretraining and Fine Tuning](images/pre_vs_fine.png)

These two stages decouple language understanding from task-specific supervision. This two-stage procedure has been remarkably effective, especially when labeled data is scarce, as the pretrained model already encodes a strong inductive bias about language.

## Why Does This Work? An Optimization View

From the perspective of training neural networks with stochastic gradient descent, pretraining provides a highly informative initialization for model parameters.

Let $\mathcal{L}_{\text{pretrain}}(\theta)$ denote the pretraining loss, and let $\hat{\theta}$ be the parameters obtained by minimizing this loss:

$$
\hat{\theta} \approx \arg\min_\theta \mathcal{L}_{\text{pretrain}}(\theta)
$$

During fine-tuning, we optimize a new task-specific loss $\mathcal{L}_{\text{finetune}}(\theta)$, starting from the pretrained parameters:

$$
\theta^* = \arg\min_\theta \mathcal{L}_{\text{finetune}}(\theta),
\quad \text{initialized at } \theta = \hat{\theta}
$$

This setup offers two complementary advantages:

1. Good starting point:  The pretrained parameters $\hat{\theta}$ already encode general linguistic knowledge, meaning that gradient-based optimization during fine-tuning is more likely to converge quickly and to a good local minimum

2. Better generalization: Due to the geometry of the loss landscape, stochastic gradient descent tends to stay relatively close to $\hat{\theta}$. If the local minima around $\hat{\theta}$ are well-aligned with generalization, then the fine-tuned model is more likely to perform well on unseen data

## Intuition Behind Gradient Propagation

Another benefit is that the gradients of the fine-tuning loss $\nabla \mathcal{L}_{\text{finetune}}(\theta)$ often propagate more effectively when $\theta$ is initialized at $\hat{\theta}$. Pretraining shapes the model’s representations such that downstream gradients flow through semantically meaningful feature spaces, improving both optimization stability and sample efficiency.

## Summary

The pretraining/fine-tuning paradigm represents a powerful instantiation of transfer learning in NLP. It leverages large-scale unsupervised data to produce models with rich linguistic priors and uses supervised fine-tuning to tailor those priors to task-specific objectives. From both empirical and theoretical perspectives, this strategy enables better generalization, faster convergence, and higher performance across nearly all areas of natural language understanding and generation.

 
## Pretraining Encoder Architectures

Transformer-based encoder models, particularly those following the BERT architecture, are pretrained using objectives that leverage the bidirectional nature of attention. Unlike autoregressive language models, which condition only on past tokens, encoder architectures benefit from full left-and-right context, enabling richer and more globally informed token representations.

## Masked Language Modeling (MLM)

The core idea behind encoder pretraining is the masked language modeling objective. In this setup, a fraction of the input tokens is replaced with a special `[MASK]` token. The model is then trained to predict the original tokens at these masked positions, conditioning on the surrounding unmasked context. Formally, given an input sequence  $x = (w_1, w_2, \dots, w_T)$  
and its corrupted version $\tilde{x}$, the model learns parameters $\theta$ that maximize the likelihood $p_\theta(x \mid \tilde{x})$.

$$
\mathbf{h}_1, \dots, \mathbf{h}_T = \text{Encoder}(w_1, \dots, w_T)
$$

$$
\hat{y}_i \sim \text{softmax}(A \mathbf{h}_i + b),
\quad \text{for masked positions } i
$$

where $A$ and $b$ are learned projection parameters.

The BERT pretraining procedure masks 15% of tokens according to the following strategy:

- 80% of the time, the token is replaced with `[MASK]`
- 10% of the time, it is replaced with a random token
- 10% of the time, it is left unchanged but still predicted

This scheme prevents the model from overfitting to the presence of `[MASK]` tokens and encourages robust representations even for unmasked inputs.

## Next Sentence Prediction (NSP)

In addition to masked language modeling, BERT was originally trained with a binary classification task known as next sentence prediction. Given two input segments, the model predicts whether the second segment follows the first in the original corpus or was randomly sampled. Subsequent studies such as RoBERTa showed that removing NSP can improve downstream performance, suggesting that NSP is not essential.

## Advancements and Variants

Numerous refinements to the BERT pretraining methodology have been proposed:

- RoBERTa:   Trains longer, on more data, removes NSP, uses dynamic masking and larger batch sizes

- SpanBERT:  Masks contiguous spans of tokens instead of individual ones, promoting span-level representations

## Limitations of Encoder-Only Pretraining

While encoder models like BERT excel at understanding and classification tasks (e.g., sentiment analysis, QA), they are not directly suited for sequence generation due to their non-autoregressive architecture. For tasks requiring fluent text generation (e.g., summarization, translation), decoder-based or encoder-decoder models are more appropriate. Nonetheless, pretrained encoders remain foundational across a wide array of NLP applications due to their strong contextual representations and adaptability to fine-tuning for downstream tasks.

 
## Pretraining Encoder–Decoder Architectures

Encoder–decoder models combine the strengths of both architectures: the encoder produces rich, bidirectional representations of input sequences, while the decoder performs autoregressive generation conditioned on these representations. This architecture is particularly well-suited for sequence-to-sequence tasks such as machine translation, summarization, and question answering.

## Why Use Encoder–Decoder Models?

- Encoders build contextual representations using bidirectional attention
- Decoders enable autoregressive generation conditioned on encoded input and past outputs
- Encoder–decoder models enable powerful conditional generation while leveraging deep understanding of the input.

## Pretraining Strategy: Language Modeling with Encoders

A naive extension of language modeling to encoder–decoder architectures splits the input sequence:

- A prefix of the input (e.g., tokens $w_1, \dots, w_T$) is passed to the encoder.
- The decoder autoregressively generates the continuation

$$
\mathbf{h}_{1:T} = \text{Encoder}(w_{1:T}), \quad
\mathbf{h}_{T+1:T'} = \text{Decoder}(w_{1:T'}, \mathbf{h}_{1:T})
$$

$$
P(y_i) \sim \text{softmax}(\mathbf{A}\mathbf{h}_i + \mathbf{b}), \quad i > T
$$

This allows the encoder to learn bidirectional features while enabling the decoder to condition on them during generation. However, more specialized objectives have shown better performance.

## Span Corruption: The T5 Objective

Span corruption is the core pretraining objective of the T5 model. Random spans of text are removed from the input and replaced with unique sentinel tokens such as `<extra_id_0>`. The decoder is trained to reconstruct the missing spans.

- Random spans of text are removed from the input
- Each removed span is replaced with a unique sentinel token (e.g., `<extra_id_0>`).
- The model is trained to reconstruct the missing spans from these placeholders.

![Span Corruption](images/span_corruption.png)

Example:

- Input: `The quick <extra_id_0> fox jumps <extra_id_1> the lazy dog.`
- Target: `<extra_id_0> brown <extra_id_1> over <extra_id_2>`

Advantages:

- The encoder benefits from full bidirectional context (unlike standard causal models).
- The decoder is trained autoregressively, generating spans conditioned on encoder output.
- The objective is implemented via input preprocessing; the model learns a language modeling task at the decoder side.
## Summary

Encoder–decoder pretraining balances bidirectional representation learning (via the encoder) with generative capability (via the decoder). Span corruption, as implemented in T5, has emerged as a highly effective strategy. It produces models that generalize well across a wide range of NLP tasks and are compatible with the "text-to-text" paradigm.

 
## Pretraining Decoder Architectures

Decoder-only language models have emerged as a central architecture in modern NLP, particularly for tasks involving natural language generation. These models are pretrained autoregressively to maximize the likelihood of the next token conditioned on previous tokens:

$$
p_\theta(w_t \mid w_{1:t-1})
$$

typically using causal (left-to-right) self-attention to ensure that each token only attends to its left context. This setup enables decoders to learn rich, context-sensitive representations suitable for both generation and classification tasks.

![Decoder Pretraining](images/decoder_pre.png)

## Representation Learning for Classification

Despite being trained for generation, pretrained decoders can be repurposed for classification tasks by leveraging their hidden representations. A common approach is to use the final token's hidden state $h_T$ as a summary of the sequence and apply a linear classifier on top:

$$
h_1, \dots, h_T = \text{Decoder}(w_1, \dots, w_T)
$$

$$
y \sim \text{softmax}(A h_T + b)
$$

where $A$ and $b$ are task-specific parameters initialized randomly and optimized during finetuning. Importantly, gradients are backpropagated through the entire decoder, enabling the model to adapt its internal representations to the downstream task.

## Sequence Generation via Pretrained Decoders

In generation settings, decoder models are used directly in the autoregressive manner in which they were pretrained. At each timestep, the decoder produces a distribution over the next token given the previously generated tokens:

$$
h_1, \dots, h_{t-1} = \text{Decoder}(w_1, \dots, w_{t-1})
$$

$$
w_t \sim \text{softmax}(A h_{t-1} + b)
$$

Here, $A$ and $b$ represent the output projection matrix and bias that were learned during pretraining. In many applications, these parameters are reused without modification; however, they may also be further finetuned alongside the decoder, depending on the task and data availability.

This generation paradigm is particularly effective for tasks such as:

- Dialogue modeling: where the decoder generates a response conditioned on the prior dialogue history.
- Summarization: where the decoder generates a summary conditioned on a document input.

## Architectural Context and Transferability

Decoder-only models differ from encoder-decoder architectures (e.g., T5) in that they do not encode the input with a separate encoder. Instead, all information is processed through the decoder itself. This simplicity enables direct reuse of the pretrained decoder for diverse tasks with minimal architectural modification.

Pretraining provides:

- Transferability: Learned representations encode rich syntactic, semantic, and factual knowledge that generalize well across tasks.
- Architectural reuse: The same pretrained model can support both classification and generation, reducing deployment complexity.
- Scalability: Autoregressive training benefits from large-scale data and scales effectively with model size.

Pretrained decoder architectures, as used in models such as GPT-2, GPT-3, and LLaMA, have demonstrated remarkable performance across a wide range of NLP benchmarks, confirming the utility of this simple yet powerful design.
