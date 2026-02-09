# Task 3: Understanding the model

## Objective
To open the "Black Box" of our best performing model (Tier C: DistilBERT + LoRA) and understand *why* it makes its decisions. By understanding the decision boundary, we aimed to craft adversarial examples that could evade detection.

## Experiment A: Interpretability with Integrated Gradients
*   **Technique**: Used `LayerIntegratedGradients` (via Captum) to visualize token-level attributions.
*   **Methodology**:
    1.  Computed the gradient of the prediction output with respect to the input embedding layer.
    2.  Integrated these gradients from a baseline (PAD token) to the actual input.
    3.  Visualized which tokens pushed the probability towards "AI" (Red) or "Human" (Green).

## Experiment B: Shuffling, Gibberish 
*   **Hypothesis**: If the model relies on *sentence structure* or *grammar*, destroying the order of words should break the classification.
*   **Methodology**:
    1.  **Full Jumble**: Randomly shuffled every word in the text.
    2.  **N-gram Shuffle**: Shuffled chunks of 2, 3, or 5 words to preserve local context.
    ### Results
*   **Jumbled Text**:  Human Text Accuracy increases to 100% and AI Text Accuracy drops by 15%. 
    - this means jumbling increased human predictions and decreased ai predictions. So is randomness = more human?

*  **Random Gibberish**: Generated completely random sequences of words using the dataset's vocabulary, to test if more random = human.  
*   **Gibberish**: Classified as **AI** with high confidence ($>99\%$), meaning there is a very specific pattern that the model is classifies as human.

### Inference
The model does *not* require coherent grammar or sentence structure to classify text. The fact that random gibberish is confidently flagged as "AI" is a strong indicator that the model defaults to "AI" in the absence of specific "Human" signals.

## Experiment C: The "Frequency" Hypothesis
*   **Hypothesis**: If word order doesn't matter, maybe it's just the *distribution* of words (e.g., how often "the" appears vs. "delve").
*   **Methodology**:
    1.  Built a frequency distribution of words for Human texts and AI texts separately.
    2.  Generated *new* random text by sampling words from these distributions (bag-of-words generation).
    3.  Fed these incoherent, distribution-matched samples to the detector.

### Results
*   **Human-Distribution Noise**: 100% classified as **Human**.
*   **AI-Distribution Noise**: 100% classified as **AI**.

### Inference
**It is inclined to more of a Bag of Words type model.** This definitively proves that the detector is essentially checking the frequency distribution of the words and not really any coherence.

## Experiment D: Vocabulary Swapping
*   **Hypothesis**: Does the model care about *structure* (sentence complexity, grammar) or just *vocabulary* (specific words)?
*   **Methodology**:
    1.  Ranked top $N$ most frequent words for both Human and AI datasets.
    2.  Created a 1:1 mapping: $Word_{AI, rank=k} \leftrightarrow Word_{Human, rank=k}$.
    3.  **Swapped Vocabulary**: Took AI texts and replaced their words with "Human" equivalents (preserving structure).
    4.  **Swapped Vocabulary**: Took Human texts and replaced their words with "AI" equivalents.

### Results (The "Bag of Words" Revelation)
*   **AI Structure + Human Vocab**: 94% classified as **Human**.
*   **Human Structure + AI Vocab**: 86% classified as **AI**.

### Inference
**Vocabulary Dominates Structure.** The sophisticated Transformer model, despite having attention mechanisms, largely degraded into a complex "Bag of Words" detector. It didn't care that the AI sentence structure was perfect; if it saw "human" words (likely emotional, informal, or typo-prone), it predicted Human.

## Experiment E: Concreteness Analysis
*   **Hypothesis**: AI text is perceived as "hollow" or "abstract". Does the model use *concreteness* as a feature?
*   **Methodology**:
    1.  Used the **Brysbaert Concreteness Ratings** dataset to score every word in our samples.
    2.  Compared the average concreteness score of correctly classified vs. misclassified samples.

### Results
*   **Correctly Classified Human**: High Concreteness Score ($1.68$).
*   **Misclassified Human (Human $\to$ AI)**: Low Concreteness Score ($1.19$).

### Inference
**Abstraction is a Proxy for AI.** When humans write about abstract concepts (philosophy, law, theory), they use language that is statistically indistinguishable from AI to this model. The model equates "Abstract/Intellectual" with "Artificial", leading to a high False Positive rate on formal human writing.

## Experiment F: Stylometric Signals
*   **Hypothesis**: Are there simple statistical proxies like Stop Word Density or Word Length that correlate with the model's predictions?
*   **Methodology**: Calculated `Stop Word Density`, `Type-Token Ratio (TTR)`, and `Avg Word Length` for datasets.

### Results
*   **Stop Word Density**: Human ($45\%$) > AI ($37\%$).
*   **Lexical Diversity (TTR)**: AI ($70\%$) > Human ($47\%$).
*   **Avg Word Length**: AI ($5.8$) > Human ($4.7$).

### Inference
AI text is "dense" with information (content words) and diverse in vocabulary. Human text is "diluted" with function words (the, a, is, of, and) and repetitive. The model likely uses this density as a strong heuristic.

## Conclusion
The model is a complex mixture of the following:
1. Choice of words
2. Concreteness 
3. Stop Word Density
4. Lexical Diversity
5. Word Length
6. Some semantics 
