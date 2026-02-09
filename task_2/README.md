# Task 2: Machine Learning Detectors

## Objective
To build and evaluate a series of progressively complex machine learning models capable of distinguishing between Human and AI-generated text. We implemented a tiered approach to observe the trade-off between conceptual simplicity and detection power.

## Tier A: The Linguist (XGBoost)
*   **Approach**: Classical Machine Learning using hand-crafted linguistic features.
*   **Model**: XGBoost Classifier (`n_estimators=200`, `max_depth=6`).
*   **Input Features**:
    1.  **Punctuation Ratio** (10.7% Importance): Density of semicolons and em-dashes.
    2.  **Hapax Legomena** (19.0% Importance): Vocabulary richness.
    3.  **Flesch-Kincaid** (18.6% Importance): Readability score.
    4.  **TTR** (16.3% Importance): Lexical diversity.
    5.  **Dependency Depth** (13.6% Importance): Syntactic complexity.
    6.  **Adj/Noun Ratio** (21.9% Importance): Descriptiveness.

(As expected from task 1)

### Performance Analysis
*   **Overall Test Accuracy**: 63.3% (Baseline: 33.3%)
*   **Confusion Matrix**:
    ```
    [[48  2  0]   <- Human (ACTUAL)
     [10 21 19]   <- AI (ACTUAL)
     [13 7 30]]   <- AI Mimic (ACTUAL)
    ```
*   **Class-Wise Performance**:
    *   **Human**: 96% Recall, 68% Precision. The model is **excellent** at identifying humans.
    *   **AI**: 42% Recall, 62% Precision. The model **fails significantly** here.
    *   **AI Mimic**: 52% Recall, 58% Precision.

* Dividing it into 3 classes, just makes this more difficult. Lets do human vs AI and just analyse the results. 

## Tier B: The Vector (Neural Network + FastText)
*   **Approach**: Shallow Deep Learning using static word embeddings.
*   **Model**: 3-Layer Feedforward Neural Network (300 $\to$ 128 $\to$ 64 $\to$ 1).
*   **Hyperparameters**: Dropout (0.3), Adam Optimizer ($lr=0.001$), BCEWithLogitsLoss.
*   **Input**: Average FastText Word Vectors (300d).
*   **Novelty**: Moved from tracking *how* it was written (syntax) to *what* was written (semantics/topics), ignoring grammar/structure (bag-of-words approach).

### Performance Analysis
*   **Test Accuracy**: 82.2%
*   **Confusion Matrix** (Reconstructed from Binary Classification):
    ```
    [[60   0]   <- Human (ACTUAL): 100% Correct
     [31  83]]  <- AI + Mimic (ACTUAL): 31 Missed, 83 Caught
    ```
    *   **Breakdown of the 31 Misses**:
        *   **Standard AI**: 4 misses (93% Accuracy)
        *   **AI Mimic**: 27 misses (50% Accuracy - Coin Flip)

    (This model is most likely overfit, since it performed very well on the training data)

## Tier C: The Specialist (Transformer + LoRA)

### 1. Architecture
*   **Base Model:** `distilbert-base-uncased` (66M parameters).
*   **Fine-Tuning Method:** **LoRA (Low-Rank Adaptation)**.
    *   Instead of retraining all weights, we freeze the base model and inject trainable rank decomposition matrices into the `query` and `value` attention projections.
    *   **Config:** Rank ($r=16$), Alpha ($\alpha=32$), Dropout ($0.1$).
    *   **Efficiency:** Only **0.9%** of parameters (~590k) were trainable, allowing for rapid iteration on consumer hardware.

### 2. Performance (The "Paranoid" Detector)
- Since this is the best detector, lets evaluate on a standard big dataset. 
Evaluated on a large-scale test set of **5,991 samples** (Mixed Human/AI).

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Overall Accuracy** | **69.26%** | Dropped significantly from the small-batch 91%. |
| **AI Recall** | **99.6%** | Catches nearly *every* AI text. |
| **Human Recall** | **38.9%** | Fails to recognize most humans. |
| **Human Precision** | **99.0%** | If it says "Human", it's definitely Human. |
| **AI Precision** | **62.0%** | Flags many humans as AI (high False Positive rate). |

### 3. Key Observations
*   **The "Paranoid Android" Effect:** The model has become hypersensitive to AI signals. It creates a "guilty until proven innocent" regime.
    *   **Low Human Recall (38.9%)**: Most human text is flagged as AI. This suggests that as we scaled up, the "Human" class in the wild became too diverse for the model's narrow training, while the "AI" signal remained consistent.
    *   **High Assurance**: The 99% Human Precision means the model *only* predicts Human when it finds a very specific signal (likely specific vocab or structural quirks like typos). All ambiguous text defaults to AI.

## Comparative Results Table

| Metric | Tier A (XGBoost) | Tier B (FastText NN) | Tier C (DistilBERT LoRA) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Symbolic/Feature-based | Static Embedding | Contextual Embedding |
| **Input** | 7 Linguistic Metrics | 300d Avg Vector | Token Sequence (256 len) |
| **Test Accuracy** | 63.3% | 82.2% | **91.2%** |
| **AI Recall** (Catch Rate) | 42.0% | 72.8% | **95.6%** |
| **AI Precision** (Trust) | 62.0% | **100.0%** | 97.7% |

## Conclusion
*   **Syntax alone is insufficient**: Tier A proved that while AI has a "style", it is subtle and easily masked.
*   **Semantics matter**: Tier B showed that AI tends to use specific vocabulary clusters (likely "safety" or "neutrality" words).
