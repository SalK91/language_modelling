# LLM evaluation

Evaluating LLMs is challenging because they generate free-form text, which cannot be fully assessed with traditional exact-match metrics. Evaluation combines human judgment, rule-based metrics, and LLM-based automated methods to measure both output quality and system performance.


## Evaluation Dimensions

### Output Quality

- Instruction Following: How well the model adheres to the user prompt or task instruction  
- Coherence: Logical consistency and fluency of generated text  
- Factuality: Accuracy and grounding of claims against known information  

### System Performance

- Latency: Time to generate responses  
- Pricing: Cost per token or query  
- Reliability: Uptime, consistency, and error rates  

## Human Evaluation

Human raters provide ground-truth quality assessments, but are subject to subjectivity. Common practices:

### Inter-Rater Reliability

- Observed agreement rate:  

    $$p_o = \frac{\text{# of times raters agree}}{\text{total instances}}$$  

- Expected agreement by chance:  

    $$p_e = \sum_{i} P(\text{rater1 chooses } i) \cdot P(\text{rater2 chooses } i)$$

- Cohen’s Kappa:  

    $$\kappa = \frac{p_o - p_e}{1 - p_e}$$ 

    - $\kappa = 1$: perfect agreement  
    - $\kappa = 0$: agreement no better than chance  

Limitations:  

- Subjective and costly  
- Cannot scale to large datasets  
- Sensitive to rating granularity (binary vs multi-class vs Likert)


## Rule-Based Metrics

Rule-based metrics compare model outputs against human references:

### BLEU (Bilingual Evaluation Understudy)

- Measures n-gram overlap between generated text and reference  
- Ranges from 0 (no overlap) to 1 (perfect match)  

$$
\text{BLEU} = \text{BP} \cdot \exp \left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

Where $p_n$ = precision of n-grams, $w_n$ = weights, BP = brevity penalty.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- Measures recall of overlapping units (n-grams, sequences, or longest common subsequence) between generated and reference text  
- Common variants: ROUGE-N (n-gram), ROUGE-L (longest common subsequence)  

Limitations:  
- Focus on surface-level matches; ignores paraphrases or semantic equivalence  
- Not sensitive to reasoning quality or factual correctness

###  METEROT (Modified Evaluation of Text Output Reliability and Truthfulness):  

$$F_{\text{mean}} (1 - p)$$  
 

## LLM-as-Judge Evaluation

Modern evaluation leverages LLMs themselves as evaluators:

- Model compares outputs against reference or prompt guidelines  
- Can provide semantic, reasoning, and factuality judgments  
- Useful for scalable evaluation when human labeling is expensive  

Example Approaches:  

1. Pairwise comparison: LLM ranks output A vs B  
2. Rubric scoring: LLM assigns scores on coherence, instruction following, factuality  
3. Fact-checking: LLM identifies factual errors or hallucinations  

Benefits:  

- Handles free-form text, paraphrases, and nuanced reasoning  
- Scalable to large datasets  

Limitations:  

- LLM may hallucinate evaluations  
- Sensitive to prompt phrasing  
- Needs calibration with human judgments for reliability


## Hybrid Evaluation Strategy

Best practice combines:

- Human evaluation for high-stakes or complex tasks  
- Rule-based metrics (BLEU, ROUGE) for surface-level quality  
- LLM-as-judge for scalable, semantic, and factual assessment  

This triangulation ensures balanced evaluation across fluency, accuracy, and reasoning, while controlling cost and scalability.


## Structured Output Evaluation

Some LLM tasks require generating structured outputs, such as JSON, XML, tables, or constrained sequences of tokens. Evaluating these outputs differs from free-form text evaluation.

### Key Considerations

- Constrained token sets: Ensure that generated tokens are valid members of the expected set (e.g., enumerated labels, IDs, or options).  
- Schema compliance: Check that outputs follow the required structure (e.g., JSON keys present, types correct).  
- Exact match vs semantic match:  
    - Exact match works well for fixed-choice outputs.  
    - Semantic match or embedding-based metrics may be used if minor variations are acceptable.  

Example:  
For a task generating a JSON object from a prompt:

```json
{
  "name": "Alice",
  "age": 30,
  "role": "Engineer"
}
```

- Exact match evaluation: Pass only if keys, types, and values exactly match reference

- Token-level evaluation: Sample from the set of allowed tokens for each field and calculate accuracy

- Schema validation: Ensure no missing keys, invalid types, or malformed structures

Structured evaluation is essential for applications that interface with APIs, databases, or downstream systems, where invalid outputs could break the workflow.