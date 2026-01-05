# Retrieval-Augmented Generation

Large Language Models (LLMs) face key limitations when used in real-world applications:

- Static knowledge: Model understanding is limited to pretraining data, which becomes outdated and cannot include private or domain-specific information.

- Context window constraints: LLMs can only process a limited number of tokens at once, making it infeasible to feed large documents directly.

- Attention dilution: Injecting long or noisy context increases the risk of hallucination, distraction, and degraded response quality.

- Token-based cost: LLM APIs charge per input and output token, making repeated ingestion of large text expensive and inefficient.


Three Stages:

1. Retrieve relevant document via similarity operation across the knowledge base. 

2. Augment prompt with relevant pieces of information.

3. Generate Response

## Retrieval
Retrieval in RAG systems is typically decomposed into candidate generation followed by ranking, each with distinct objectives and design choices.

### 1. Candidate Retrieval (Recall-Oriented)

Goal: Maximize recall by selecting a superset of potentially relevant document chunks.

Key hyperparameters:
- Embedding dimension
- Chunk size
- Chunk overlap

Retrieval techniques:

- Dense semantic embeddings
- Sparse keyword-based retrieval
- Or a combination of both

Methods:

- Semantic Retrieval: Embedding-based similarity search (e.g., cosine similarity)
- Keyword Retrieval: Lexical matching using BM25
- Hybrid Retrieval: Weighted or learned combination of dense and sparse signals


### 2. Ranking / Re-ranking (Precision-Oriented)

Goal: Maximize precision by assigning accurate relevance scores to a smaller candidate set.

- Apply re-ranking models over top-$k$ retrieved candidates
- Use cross-encoders or LLM-based scorers for deeper query–document interaction
- Produce a final relevance score used for context selection

> bi vs cross encoders: Bi-encoder encodes query and document independently into vectors → fast, scalable, supports ANN search, optimized for recall. Cross-encoder jointly processes query + document with full attention → slower, runs only on small candidate sets, optimized for precision and re-ranking. In RAG: Bi-encoder retrieves, Cross-encoder re-ranks.

### 3. Embedding Discrepancy Mitigation

Dense embeddings may suffer from semantic drift or underspecification.

Mitigation strategies:

- Create “fake documents”: Generate a fake document based on prompt and embedd the fake document instead of the prompt.

- Contextualize chunks by prepending:
    - Document titles
    - Section headers
    - Metadata

- Encode chunks in a way that preserves global document intent


### 4. Retrieval Evaluation Metrics

Retrieval quality is quantified independently of generation using standard IR metrics:

- Recall@K:
  $$
  R@K = \frac{\text{relevant retrieved at } K}{\text{total relevant}}
  $$

- Precision@K:
  $$
  P@K = \frac{\text{relevant retrieved at } K}{K}
  $$

- Reciprocal Rank (RR):
  $$
  RR = \frac{1}{\text{rank of first relevant result}}
  $$

- Normalized Discounted Cumulative Gain (NDCG@K):
  $$
  DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}
  $$

    where  $rel_i \in {0,1}$
  $$
  NDCG@K = \frac{DCG@K}{IDCG@K}
  $$

    where $IDCG@K$ is Ideal $DCG@K$ i.e. if ranking was perfect.

These metrics capture ranking quality, early relevance, and coverage, and are critical for diagnosing downstream RAG performance.
