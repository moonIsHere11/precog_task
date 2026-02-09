# Task 4: The Breach (Genetic Algorithm Evasion)

## 1. Objective
To automate the evasion of our best-performing AI detector (Tier C: DistilBERT + LoRA) using a Genetic Algorithm (GA). The goal was to take text classified as "AI" ($P(Human) \approx 0$) and iteratively mutate it until it passed as "Human" ($P(Human) > 0.9$), without losing its original meaning.

## 2. The Thought Process & Strategy
In Task 3 ("The Ghost"), we discovered that our detector was essentially a sophisticated "Bag of Words" classifier heavily biased by specific vocabulary. We didn't want to blindly mutate the text; we wanted to use these insights to guide our attack.

Here is how we translated our initial insights into specific engineering solutions:

### Problem 1: The "Right" Words Matter
*   **Insight**: The model is chemically dependent on specific "human" words. Random synonym swapping isn't enough; we need the wide variety of words that appeared in our human training corpus but *never* in the AI corpus.
*   **Solution**: We constructed a **Human Vocabulary Bank** from the Task 3 dataset (6,721 unique words found in Human texts but absent in AI texts).
*   **Implementation**: Instead of randomly picking words, we used `spaCy` word vectors to dynamically select the most *contextually relevant* human words for each specific input paragraph. We then injected these words directly into the LLM prompt as a "suggested vocabulary list".

### Problem 2: Semantic Drift
*   **Insight**: An unconstrained LLM might rewrite "The cat sat on the mat" as "The feline rested on the rug" (too formal/abstract) or completely change the meaning to satisfy the classifier.
*   **Solution**: We incorporated **Semantic Similarity** directly into the Fitness Function as a penalty term.
*   **Implementation**:
    $$ F = P(Human) - \lambda \cdot (1 - \text{Similarity}) $$
    *   We tuned $\lambda$ (initially 0.05, increased to 0.15) to heavily penalize any mutation that drifted too far from the original meaning. This ensured the result was still the *same* paragraph, just "humanized".

### Problem 3: The "Concreteness" & "Simplification" Bias
*   **Insight**: Task 3 showed the model associates "Abstract" and "Complex" language with AI. To pass as human, we need to be concrete and simple.
*   **Solution**: We hard-coded specific stylistic constraints into the system prompt for the mutation engine (Gemini).
*   **Implementation**: The prompt included explicit instructions:
    1.  "Use smaller words (word length $\approx$ 4.5 characters)"
    2.  "Slightly higher stop words ratio"
    3.  "Lower Type-Token Ratio (TTR)"
    4.  "Replace abstract words with their more concrete versions"

## 3. The Implementation (Genetic Algorithm)
We built a Genetic Algorithm with the following components:

1.  **Parent Selection**: The original AI-generated paragraph.
2.  **Mutation Operator**: An LLM (Gemini 2.5 Flash) tasked with rewriting the paragraph using the injected "Human Vocabulary" and adhering to the stylistic constraints.
3.  **Fitness Evaluation**:
    *   **Detector Score**: Output from our Fine-Tuned DistilBERT model.
    *   **Semantic Score**: Cosine similarity from `sentence-transformers` (`all-MiniLM-L6-v2`).
4.  **Survival of the Fittest**: The single best variant becomes the parent for the next generation.

## 4. Results
We ran the full pipeline on a batch of 20 diverse AI-generated paragraphs.

### 4.1 Success Rate: 90%
*   (18/20) successfully tricked the detector into classifying it as "Human" with $>90\%$ confidence.

### 4.2 Speed & Efficiency
*   **Average Iterations**: $2.3$
*   Most paragraphs were converted in a 2 iterations. The "Human Vocabulary Injection" was so effective that the model often flipped immediately from $P(Human)=0.01$ to $P(Human)>0.8 in one iteration.$.

### 4.3 Quality Preservation
*   **Average Semantic Similarity**: $\approx 0.80$
*   The rewritten texts remained coherent and faithful to the original message, proving that "style" can be manipulated independently of "content".

## Conclusion
We have successfully breached the defense. By systematically exploiting the specific weaknesses identified in the "Glass Box" analysis (Task 3)—specifically vocabulary bias and abstraction sensitivity—we created an automated system that bypasses the AI detector system. 
