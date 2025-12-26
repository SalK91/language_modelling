# Chapter 7: Evaluation Metrics in NLP

Evaluating NLP models is essential to ensure their performance, generalizability, and utility. The appropriate metric depends on the specific task, data characteristics, and desired outcome. Below, we detail evaluation methods used for key NLP tasks and provide a comparative table of metrics.

Evaluation metrics quantify different aspects of model behavior, such as accuracy, fluency, semantic adequacy, and robustness, and no single metric is sufficient for all tasks.


## Machine Translation Evaluation
Machine translation is a structured generation task where outputs are compared against one or more human reference translations.

### 1. BLEU (Bilingual Evaluation Understudy)

BLEU is a precision-based metric used to evaluate the quality of machine-translated text by comparing it to one or more human reference translations. It uses modified n-gram precision and a brevity penalty to penalize overly short translations.

$$
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

where:

- $p_n$ is the modified (clipped) precision for n-grams of size $n$
- $w_n$ is the weight for each n-gram order (typically $w_n = \frac{1}{N}$)
- $\text{BP}$ is the brevity penalt
y to penalize translations that are too short
BLEU primarily measures surface-level n-gram overlap and does not directly capture semantic adequacy or fluency.

The clipped precision $p_n$ is defined as:

$$
p_n =
\frac{
\sum_{\text{ngram} \in C}
\min\left(
\text{Count}_{\text{clip}}(\text{ngram}),
\text{MaxRefCount}(\text{ngram})
\right)
}{
\sum_{\text{ngram} \in C}
\text{Count}(\text{ngram})
}
$$

where:

- $C$ is the set of n-grams of order $n$ in the candidate translation
- $\text{Count}(\text{ngram})$ is the number of times an n-gram appears in the candidate
- $\text{MaxRefCount}(\text{ngram})$ is the maximum number of times the n-gram appears in any reference translation
- $\text{Count}_{\text{clip}}(\text{ngram})$ is clipped to avoid overcounting spurious matches

Clipping prevents a system from gaining excessive credit by repeating n-grams that appear only a limited number of times in the references.

The brevity penalty (BP) is given by:

$$
\text{BP} =
\begin{cases}
1 & \text{if } c > r \\
\exp\left(1 - \frac{r}{c}\right) & \text{if } c \le r
\end{cases}
$$

where:

- $c$ is the length of the candidate translation
- $r$ is the effective reference length (usually the closest reference length to $c$)

Without the brevity penalty, a system could achieve high precision by generating unnaturally short translations.

Use: BLEU is widely used for evaluating machine translation systems and is more reliable at the corpus level than at the sentence level.



### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Unlike BLEU, which emphasizes precision, ROUGE is recall-oriented and focuses on how much of the reference content is covered by the generated text.

Originally designed for summarization, ROUGE can also be used for machine translation and other generation tasks. It primarily measures recall of overlapping units such as n-grams and longest common subsequences.

- ROUGE-N: Overlap of n-grams (e.g., ROUGE-1, ROUGE-2)
- ROUGE-L: Longest Common Subsequence (LCS)-based recall
- ROUGE-S: Skip bigram-based F-measure

$$
\text{ROUGE-N Recall} =
\frac{
\sum_{\text{reference}}
\text{match}_{n\text{-grams}}
}{
\sum_{\text{reference}}
\text{total}_{n\text{-grams}}
}
$$

Use: Especially useful in summarization tasks and adaptable to translation.

### Other Metrics for MT and Generation

Recent evaluation metrics aim to move beyond surface overlap by incorporating linguistic knowledge or learned semantic representations.

- METEOR: Considers synonym matching, stemming, and alignment; combines precision and recall using an F-score
- chrF: Character n-gram F-score; language-agnostic and effective for morphologically rich languages
- BERTScore: Uses contextual embeddings from BERT to compute similarity between candidate and reference
- COMET / BLEURT: Learned metrics trained on human judgment data; state-of-the-art for semantic machine translation evaluation

While these metrics correlate better with human judgments, they depend on pretrained models and training data biases.

## Comprehensive Evaluation Table for NLP Tasks

Table below summarizes commonly used metrics across major NLP tasks, highlighting typical evaluation practices rather than exhaustive standards.

| Task | Common Metrics | Notes |
|------|----------------|-------|
| Part-of-Speech Tagging | Accuracy | Fraction of correctly predicted tags per token |
| Named Entity Recognition | Precision, Recall, F1-score (token/span-level) | Evaluates boundary and entity type correctness using strict span or loose token overlap |
| Sentiment Analysis | Accuracy, F1-score, MCC | Binary or multi-class classification; class imbalance is often a concern |
| Text Classification | Accuracy, Precision, Recall, F1-score | General-purpose classification; often multi-label or hierarchical |
| Machine Translation | BLEU, ROUGE, METEOR, chrF, COMET, BERTScore | BLEU is standard; newer metrics better reflect semantic adequacy |
| Summarization | ROUGE-1/2/L, BERTScore, BLEURT | ROUGE-L captures sequence overlap; embedding-based metrics assess meaning |
| Question Answering | EM, F1-score | Used for span-based QA; F1 accounts for partial overlap |
| Text Generation | BLEU, ROUGE, BERTScore, MAUVE, Human evaluation | Open-ended generation requires fluency and diversity metrics |
| Dialogue Systems | USR, BLEU, METEOR, BERTScore, Human evaluation | Measures relevance, coherence, and engagement |
| Text Simplification | SARI, FKGL, BLEU | SARI evaluates beneficial edits; BLEU may not correlate with simplicity |
| Coreference Resolution | MUC, B$^3$, CEAF | Measures clustering of entity mentions; metrics are often used together |
| Language Modeling | Perplexity, Cross-Entropy | Evaluates next-token prediction; lower is better |
| Paraphrase Detection / Text Similarity | Accuracy, Cosine similarity, Pearson/Spearman correlation | Used in semantic textual similarity and entailment tasks |
| Information Retrieval / Ranking | MRR, MAP, Recall@k, nDCG | Measures ranking quality for documents or passages |



In practice, reliable evaluation often combines automatic metrics with human judgment, especially for open-ended generation tasks. Metric selection should be guided by task objectives, data characteristics, and known limitations of each evaluation method.