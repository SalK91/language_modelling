# Reasoning

## What Do We Mean by “Reasoning”?

Reasoning is the disciplined use of facts and logic to reach new conclusions.

- Deductive reasoning derives conclusions that must be true if the premises are true.
- Inductive reasoning generalises from repeated observations to predict what is likely to hold in future.
- Abductive reasoning infers the most plausible explanation for an observation (“inference to the best explanation”).

## Reasoning in Large Language Models (LLMs)

Large language models excel at predicting the next token given a context — essentially a massively multivariate pattern-completion task. Whether this *is* reasoning is an open research question, but several techniques reliably elicit reasoning-like behavior.

### Prompt Engineering

- Chain-of-Thought (CoT) prompts append reasoning demonstrations (e.g., "Let's think step-by-step") to encourage the model to reveal latent intermediate states. 
- Least-to-Most (LtM) decomposition breaks a hard task into a sequence of smaller sub‑problems; the LLM solves each sub‑problem in order, reducing error accumulation. 

### Counterfactual Probes

By editing premises and comparing completions, researchers can distinguish memorisation from real generalisation: if the model’s answer tracks the \emph{counterfactual} change, it has formed some causal abstraction rather than merely retrieving training data.

### Limitations

- Current LLMs have no explicit logical machinery or world model; apparent ``logic`` emerges from statistical correlations. 

- Faithfulness is not guaranteed: the model may generate fluent but incorrect reasoning chains (hallucinations).

-  Working memory is finite, so long multi‑step proofs can overflow the context window unless external scratchpads or tool use are provided.
