# Pre-training
Pretraining refers to learning general-purpose language representations from large amounts of unlabeled text before adapting models to specific tasks. This approach decoupled lexical representation learning from task-specific modeling.

## From Pretrained Embeddings to Pretrained Models

Before the rise of large-scale language models, pretraining in NLP primarily focused on static word embeddings such as Word2Vec, GloVe, or FastText. These embeddings provided fixed, context-independent vector representations for each word in the vocabulary, which were then used as input features for downstream models.

### Circa 2017: Word embeddings + contextual models
A common pipeline during this period combined pretrained embeddings with supervised task-specific models.

- Begin with pretrained word embeddings (context-agnostic)
- Learn contextualization during supervised training on a downstream task (e.g., sentiment analysis, question answering)
- Recurrent architectures (e.g., LSTMs) or early Transformer variants were used to build task-specific context
- Limitations:
    - Downstream data must be large and diverse enough to teach the model language understanding
    - Most model parameters are randomly initialized and trained from scratch
    - Inefficient reuse of linguistic knowledge across tasks

As a result, much of the burden of language understanding was placed on downstream supervision.

![Pretrained Embeddings](images/pre_old.png)

## Modern Paradigm: Pretraining Whole Models

In modern NLP systems, the dominant paradigm has shifted from pretraining isolated components (e.g., word embeddings) to pretraining entire models on unsupervised or self-supervised objectives over massive corpora. These pretrained models are then fine-tuned on specific tasks, often with limited labeled data.

Key characteristics of whole-model pretraining

- Nearly all model parameters are initialized using large-scale pretraining
- The model is trained on unlabeled text by corrupting the input and optimizing for its reconstruction. 
- Common objectives: Different pretraining objectives reflect different modeling assumptions and downstream use cases.
    - Masked language modeling (MLM): Randomly mask tokens and train the model to predict them (e.g., BERT)
    - Causal language modeling (CLM): Predict the next token given previous ones (e.g., GPT)
    - Permutation-based objectives: Learn to reason over non-sequential token orders (e.g., XLNet)
- Models learn:
    - Deep, contextualized representations of language structure and semantics
    - Strong priors for downstream task learning via transfer
    - Coherent probability distributions over sequences, enabling sampling and generation. These learned representations can be rapidly adapted to new tasks through fine-tuning or prompting.

## Benefits of Pretraining
Whole-model pretraining offers several practical and theoretical advantages.

- Data efficiency: Fine-tuning requires significantly fewer task-specific labeled examples
- Robust generalization: Pretrained models capture syntactic, semantic, and world knowledge
- Unified architecture: A single pretrained backbone can be reused across a wide range of tasks, reducing task-specific engineering
- Sampling and generation: Language models trained with causal objectives can be used to generate fluent, coherent text

![Pretraining](images/pre_new.png)

## Summary

Pretraining has revolutionized NLP by shifting the burden of language understanding from downstream task data to large-scale, general-purpose models. This transition from static embeddings to pretrained Transformers has enabled rapid progress across virtually every NLP benchmark, making pretraining the foundation of modern language understanding and generation systems.
