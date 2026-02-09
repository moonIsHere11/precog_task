# Task 1: Lexical & Syntactic Analysis

## Objective
The goal of this task was to move beyond subjective "vibes" and mathematically prove that human writing (Class 1), AI-generated text (Class 2), and AI-mimicry (Class 3) constitute distinct stylistic classes. Aimed to identify a set of linguistic features that consistently separate these classes regardless of topic.

## Feature Engineering: Measuring the "Style"
Extracted a set of metrics capturing different dimensions of writing style. 

### 1. Lexical Richness & Diversity
*   **Windowed Type-Token Ratio (TTR)**:
    *   *Why*: Standard TTR is biased by text length. We used a sliding window (N tokens at once) to calculate purely local vocabulary diversity.
    *   *Finding*: Surprisingly, AI models (Class 2 & 3) exhibited slightly higher mean TTR scores (~0.42-0.44) compared to the Human baseline (~0.40). This suggests AI may artificially inflate vocabulary diversity or lack the thematic repetition common in focused human argumentation.
    * Tried looking at the shift of TTR through the document as well as the density of chunks in a specific TTR range for a specific document, which revealed small but noticeable difference in distributions, band sizes etc. (Plots in notebook)
*   **Hapax Legomena Ratio**:
    *   *Why*: The ratio of words that appear only once. This proxies for the depth of vocabulary and creativity.
*   **Adjective/Noun Ratio**:
    *   *Why*: Captures the "descriptiveness" vs. "information density" of the text. Does either of the classes over describe the same stuff. 
    *   *Implementation*: Normalized counts using spaCy POS taggers.
    * Saw a Noticable bump in values for AI written texts. 

    | Metric | Class 1 (Human) | Class 2 (AI) | Class 3 (AI Mimic) |
    | :--- | :--- | :--- | :--- |
    | **Mean Windowed TTR** | 0.4023 | 0.4289 | 0.4478 |
    | **TTR Std Dev** | 0.0216 | 0.0357 | 0.0251 |
    | **Hapax Legomena Ratio** | 0.1515 | 0.14 | 0.1546 |
    | **Adj/Noun Ratio** | 	0.4059 | 0.4630| 0.4908 |

### 2. Syntactic Complexity
*   **Average Dependency Tree Depth**:
    *   *Why*: Uses spaCy's dependency parser to measure sentence nesting and structural complexity.
    *   Human texts (especially 19th-century philosophers like Mill and Russell) often use highly recursive, deep sentence structures that modern AI tends to simplify for readability. (Confirmed by PCA separation). 

### 4. Rhythm & Punctuation
*   **Punctuation Density (per 1000 tokens)**:
    *   *Features*: Semicolons, Em-dashes, Exclamation marks.
    *   *Why*: Punctuation correlates strongly with authorship. Humans use distinct punctuation (e.g., extensive semicolon usage in classical texts) to control pacing, which AI frequently specifically normalizes to modern web standards.

### 5. Readability
*   **Flesch-Kincaid Grade Level**:
    *   *Why*: Provides a composite metric of sentence length and word syllabic length.
    * As expected, AI mimic comes out on top for this field, because of the prompt instructions inclining it to be complex and formal (to mimic the author). 

| Metric | Class 1 (Human) | Class 2 (AI) | Class 3 (AI Mimic) |
| :--- | :--- | :--- | :--- |
| **Avg Dependency Depth** | 7.03 | 6.52 | 6.85 |
| **Semicolon Density (per 1000 tokens)** | 6.49 | 1.42 | 1.41 |
| **Em-Dash Density (per 1000 tokens)** | 0.00 | 2.47 | 0.60 |
| **Flesch-Kincaid Grade** | 15.52 | 16.49 | 17.79 |

### Data Processing
1.  **Normalization**: Since files varied significantly in length, all frequency metrics (punctuation) were normalized per 1000 tokens.
2.  **Handling Short Docs**: A roadblock encountered was documents shorter than our analysis window (500 tokens). These were identified (Row 16 in our dataset) and excluded to prevent NaN propagation in statistical models.
3.  **Standardization**: Features had vastly different scales (e.g., TTR is 0-1, Dependency Depth is 0-10, Punctuation is 0-50). We applied `StandardScaler` ($z = (x - u) / s$) before dimensionality reduction to ensure no single feature dominated the variance.

## Statistical Proof of Distinctness

### Dimensionality Reduction (PCA)
We applied Principal Component Analysis (PCA) to project this 9-dimensional feature space into 2D and 3D for visualization.

*   **Variance Explained**: The first 3 Principal Components captured **~85.6%** of the total variance in the dataset.
*   **Visual Separation**:
    *   **PC1 (40% variance)** strongly separated Class 1 (Human) from Class 2 (AI). This vector likely encodes "archaic complexity" (high semicolon/em-dash usage, deep syntax).
    *   **PC2 (35% variance)** helped differentiate the AI-mimic text (Class 3) from pure AI, suggesting that detailed prompting successfully shifts the style style vector, but not enough to perfectly overlap with humans.

* Variance itself is not sufficient to show the seperation of classes, all it depicts is the amount of original information that can be represented using the new set of Principal Axes. Thats why we must calculate where the centroids are.  

### Centroid Analysis
We calculated the centroids of each class in this reduced 3D space:
*   **Class 1 (Human)**: PC1 [-2.36], PC2 [3.09] — *Highly distinct region.*
*   **Class 2 (AI)**: PC1 [-0.13], PC2 [-1.10] — *Opposite quadrant.*
*   **Class 3 (Mimic)**: PC1 [1.19], PC2 [-0.15] — *Shifts towards Human on PC1 but remains distinct on PC2.*

## Quantitative Results Summary

| Metric | Class 1 (Human) | Class 2 (AI) | Class 3 (AI Mimic) |
| :--- | :--- | :--- | :--- |
| **PCA Centroid (PC1)** | -2.36 | -0.13 | 1.19 |
| **PCA Centroid (PC2)** | 3.09 | -1.10 | -0.15 |
| **PCA Centroid (PC3)** | -0.10 | 0.18 | -0.16 |

## Conclusion
Task 1 successfully proved that **Generative AI has a quantifiable "accent"**. Even when prompted to mimic specific styles (Class 3), the underlying syntactic and lexical distributions remain statistically distinguishable from the original human authors.
