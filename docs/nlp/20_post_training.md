# Post-trainings: Prompting, Instruction Finetuning, and DPO/RLHF

## Zero-Shot and Few-Shot In-Context Learning

### Zero-Shot In-Context Learning
Large language models (LLMs), beginning with GPT-2 and especially prominent in GPT-3 and beyond, exhibit a remarkable emergent ability: zero-shot learning. In this setting, a model can perform a task without any task-specific gradient updates or labeled examples—just by conditioning on the right input format.

#### Example (QA as Language Modeling):
> Passage: Tom Brady is an American football player born in San Mateo, California.  
> Q: Where was Tom Brady born? A:

The model simply continues the sequence by generating the most probable completion (“San Mateo, California”).

#### Example (Winograd Schema):
The model can also answer commonsense questions by comparing the likelihood of different completions:
> The cat couldn't fit into the hat because it was too big.

Here, the model disambiguates the pronoun “it” by comparing:
$$
P(\text{...because the cat was too big}) \quad \text{vs.} \quad P(\text{...because the hat was too big})
$$
A higher probability assigned to the second option indicates the model's preference for “hat” as the correct referent, aligning with human intuition.

### Few-Shot (In-Context) Learning
In few-shot in-context learning, the model is given a small number of input-output examples as context, followed by a new input to predict. No parameter updates occur—learning happens purely via conditioning on the prompt.

#### Example (Sentiment Classification):
> Review: This movie was amazing. Sentiment: Positive  
> Review: The plot was dull and slow. Sentiment: Negative  
> Review: The acting was brilliant. Sentiment:

The model is expected to complete the pattern with `Positive`.

#### Terminology Note:
This is often referred to as in-context learning (ICL) to distinguish it from traditional few-shot learning involving optimization-based methods.

#### Limitations of Prompting
Despite the versatility of prompting, in-context learning has limitations:

- Context window size: Prompt length is bounded by the model's maximum context window, limiting how many examples or instructions can be included.
- Reasoning complexity: Some tasks, especially those involving multi-hop or logical reasoning, remain difficult even for very large models using sophisticated prompts.
- Interpretability and control: Prompting alone provides limited control over internal reasoning steps or modular behavior.

#### Summary
- In-context learning enables task generalization without gradient updates.
- Prompt design (e.g., Chain-of-Thought, few-shot examples) significantly influences performance.
- However, for complex reasoning tasks, prompting may not be sufficient—gradient-based finetuning or hybrid techniques (e.g., tool use, scratchpads) may be necessary.

### Instruction Finetuning

Instruction finetuning is a paradigm in which a pretrained language model is adapted using a diverse set of (instruction, output) pairs across a wide range of natural language tasks. The goal is to teach the model to follow explicit instructions phrased in natural language, enabling it to generalize to unseen tasks at inference time without further finetuning.

#### Core Methodology
The core idea is to treat language modeling as a multitask learning problem, where each task is presented as a textual instruction, and the model is finetuned to produce the appropriate output. This process involves:

- Collecting a large, heterogeneous corpus of $(\text{instruction}, \text{response})$ pairs spanning tasks such as question answering, summarization, classification, translation, commonsense reasoning, etc.
- Optionally applying task mixtures (e.g., T0) or task reformulations to promote robustness.
- Finetuning the model using a standard language modeling objective on these paired samples.

#### Benefits
- Simplicity: A unified finetuning pipeline can teach the model to perform a wide range of tasks, framed uniformly as instruction following.
- Generalization: Models trained this way can often generalize to novel tasks simply by being given a well-phrased prompt—even tasks not seen during training (zero-shot generalization).
- Improved usability: End-users do not need to know the model’s internal API or finetune further; they can interact with the model via natural language instructions.

#### Challenges and Limitations
- Data cost: Collecting high-quality demonstrations for hundreds of diverse tasks is resource-intensive and often requires human annotation.
- Alignment mismatch: The model is optimized for a maximum likelihood objective, which may not align with human preferences (e.g., helpfulness, harmlessness, or truthfulness), leading to outputs that are technically correct but pragmatically unhelpful.
- Ambiguity in instructions: Natural language instructions can be underspecified or ambiguous, requiring either better dataset design or model mechanisms for disambiguation.

#### Representative Models
Instruction finetuning has been a foundational technique behind several recent LLMs:

- T5 (Text-to-Text Transfer Transformer): Casts all tasks into a text-to-text format with task-specific prefixes (e.g., “translate English to German: ...”).
- FLAN (Finetuned Language Net): Builds on T5 with instruction finetuning across many tasks and task variants.
- T0: Trains on prompt collections to enable strong zero-shot task generalization.
- InstructGPT: Combines instruction finetuning with reinforcement learning from human feedback (RLHF) to better align with human intent.

#### Outlook
Instruction finetuning plays a key role in aligning language models with human goals and making them more broadly useful without requiring users to craft task-specific prompts. However, as models scale and tasks become more complex, instruction finetuning alone may be insufficient, motivating hybrid approaches that integrate tool use, memory, and interaction (e.g., agents or retrievers).


### Optimizing for Human Preferences (RLHF / DPO)

While instruction finetuning significantly improves a language model's usability by teaching it to follow task instructions, it does not guarantee alignment with human values, preferences, or expectations. To bridge this gap, researchers have developed techniques that explicitly optimize language models based on human preferences. Two leading approaches in this domain are Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

#### From Instructions to Rewards
Instruction tuning is typically the first step in building an aligned model. However, aligning model outputs more precisely with human expectations requires further optimization:

- Step 1: Instruction finetuning on diverse (instruction, output) pairs.
- Step 2: Learning a reward function from human feedback.
- Step 3: Optimizing the model to produce outputs that maximize this learned reward.

![RLHF](images/RLHF.png)


#### Learning Human Preferences
Key challenge: Obtaining human feedback at scale is expensive, inconsistent, and noisy.

Solution: Instead of relying on absolute scores or direct ratings—which are often uncalibrated—use pairwise comparisons, where humans are asked to choose the preferred output among two or more options. These comparisons can then be modeled as a binary classification task.

A commonly used model for interpreting such data is the Bradley–Terry model, which assigns a higher latent score to the preferred (“winning”) output over the non-preferred (“losing”) one.


#### Recipe to get (pairwise) preference data

1. Generate a pair of responses $(\hat{y}_1, \hat{y}_2)$ for the same prompt $x$
    - Input $x$ via logs / reference distribution  
    - Output $\hat{y}$ via SFT model with $T > 0$ (synthetic / rewrites / sampling)

2. Label $(x, \hat{y}_1)$ and $(x, \hat{y}_2)$

    - Human rating  
    - Proxies (e.g., LLM-as-a-judge, BLEU, ROUGE, etc.)  
    - Variants:
        - Binary scale: $1$ if better, $0$ if worse  
        - Nuanced scale: e.g., $s \in [0, 1]$ or Likert-style ordinal scores

#### Reinforcement Learning from Human Feedback (RLHF)

1. Reward Modeling: Distinguish good from bad! 
    Collect human preference data in the form of comparisons between model outputs. Train a reward model (RM) to predict human preferences by learning to assign higher scores to preferred responses.

    - Input: (prompt $x$, response $\hat{y}$)
    - Output: quantitative score $r(x,\hat{y})$
    - Learn $r$ based on pairwise preference data (Bradley-Terry formulation)

2. Reinforcement learning: Align the model!
    Use reinforcement learning (typically Proximal Policy Optimization, PPO) to optimize the language model to produce outputs that maximize the predicted reward, while remaining close to the supervised model (e.g., via a KL-divergence penalty).

Strengths:

- Produces high-quality, aligned outputs when tuned well.
- Flexible reward modeling enables fine-grained preference learning.

Challenges:

- RL optimization is unstable, sensitive to reward shaping and hyperparameters.
- Computationally expensive: PPO with large models requires extensive resources.
- Human preferences are not always consistent, and reward models may overfit artifacts in the training data.

#### Direct Preference Optimization (DPO)
DPO offers a simpler, stable alternative to RLHF by bypassing the need for reinforcement learning altogether. Instead, it directly finetunes the language model on human preference data using a contrastive objective.

Given a dataset of preferred (“chosen”) and non-preferred (“rejected”) outputs, the model is trained to increase the likelihood of preferred completions relative to rejected ones.

Advantages:

- No reinforcement learning loop — avoids instability and simplifies training.
- Leverages the pretrained model’s probabilities directly.
- Comparable performance to RLHF on many benchmarks.

Limitations:

- Assumes high-quality, diverse comparison data.
- Does not inherently model long-term or interactive objectives.

#### Summary

- Optimizing for human preferences is essential for building safe, aligned LLMs.
- RLHF has shown strong results (e.g., InstructGPT, ChatGPT), but is computationally intensive.
- DPO is emerging as a lightweight, scalable alternative with competitive results.
- Both methods rely on accurate human preference modeling—an inherently noisy and subjective signal.

| Term | Description |
|------|-------------|
| Instruction Fine-Tuning (IFT) | Training a language model to follow user instructions, typically using an autoregressive language modeling loss. |
| Supervised Fine-Tuning (SFT) | Training a model on task-specific labeled data to acquire desired capabilities, generally with an autoregressive LM loss. |
| Alignment | A broad goal of making models behave according to user intentions or societal values; can be optimized via any suitable loss function. |
| Reinforcement Learning from Human Feedback (RLHF) | A specific technique that fine-tunes models using human-generated preference data via reinforcement learning. |
| Preference Fine-Tuning | Uses human-labeled preference data to fine-tune models. Can be implemented via RLHF, Direct Preference Optimization (DPO), or learning-to-rank methods. |
