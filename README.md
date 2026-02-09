# PreCog Task: AI Text Detection & Analysis

> **Disclaimer**: This README provides a high-level overview of the entire project. For detailed methodology, code, and results, please refer to the individual README files for each task.

This repository contains a comprehensive study on detecting and analyzing AI-generated text, exploring stylistic differences between human writing, AI-generated content, and AI-mimicry.

## Project Overview

The project is divided into five main tasks, each building upon insights from the previous:

0. **Dataset Creation** - Curating high-quality human and AI-generated text datasets
1. **Lexical & Syntactic Analysis** - Mathematical proof that AI has a quantifiable "accent"
2. **Classification Models** - Building robust detectors for AI-generated text
3. **Model Interpretability & Analysis** - Understanding what makes text "detectable"
4. **Bypass Model** - Using genetic algorithms to bypass detection

---

## Tasks

### Task 0: Dataset Creation

**Goal**: Create a robust, topic-controlled dataset with three classes: Human writing (Mill & Russell), AI-generated text (Gemini), and AI-mimicry.

**Approach**:
- **Class 1**: Cleaned philosophical texts (100-200 word paragraphs)
- **Class 2**: AI essays generated on 11 topics extracted via BERTopic
- **Class 3**: AI mimicking author styles using extracted stylistic features

[Link to README of Task 0](./datasets/README.md)

---

### Task 1: Lexical & Syntactic Analysis

**Goal**: Mathematically prove that human writing, AI-generated text, and AI-mimicry constitute distinct stylistic classes.

**Key Findings**:
- Extracted 9 linguistic features capturing lexical richness, syntactic complexity, rhythm, and readability
- PCA analysis showed clear separation between classes (85.6% variance explained)
- AI models exhibit higher TTR but lower dependency depth compared to human texts

 [Link to README of Task 1](./task_1/README.md)

---

### Task 2: Classification Models

**Goal**: Build and evaluate classification models to detect AI-generated text.

#### Tier A: XGBoost Classifier 
- Stylistic Features of Text 
- 63% accuracy (across 3 classes)

#### Tier B: Neural Network Classifier
- Feedforward neural network with embeddings
- Achieved 81.2% accuracy on test set

#### Tier C: DistilBERT + LoRA
- Fine-tuned DistilBERT with Low-Rank Adaptation
- Accuracy of 91.2% 

 [Link to README of Task 2](./task_2/README.md)

---

### Task 3: Model Interpretability & Analysis

**Goal**: Understand *what* the model is basing its predictions off.

**Analysis Performed**:
1. Token Attribution (Integrated Gradients)
2. Stylometric Features validation
3. Concreteness Analysis (abstract vs concrete words)
4. Structural Analysis 

**Key Insights**:
- Misclassified human texts are more abstract (concreteness: 1.68 vs 1.56)
- Model relies heavily on vocabulary differences
- Function words surprisingly important for classification

 [Link to README of Task 3](./task_3/README.md)

---

### Task 4: Adversarial Attacks via Genetic Algorithm

**Goal**: Use insights from Task 3 to iteratively modify AI text to bypass the classifier.

**Approach**:
- Built human-exclusive vocabulary (~6000 words)
- Used Gemini API with stylometric guidance
- Fitness: `F = P(Human) - λ × (1 - semantic_similarity)`

**Results** (20 AI paragraphs):
- **Conversion rate**: 90% (18/20)
- **Average iterations**: 2.1
- **Semantic similarity**: 0.78

**Takeaway**: Current classifier are vulnerable to informed adversarial attacks.

 [Link to README of Task 4](./task_4/README.md)

---

## Key Technologies

- **NLP**: spaCy, NLTK, Transformers (Hugging Face)
- **Deep Learning**: PyTorch, DistilBERT, LoRA (PEFT)
- **ML Libraries**: scikit-learn, XGBoost, sentence-transformers
- **Interpretability**: Captum (Integrated Gradients)
- **LLM API**: google.genai 
- **Visualization**: matplotlib, plotly
- **Misc**: NumPy, Pandas, Pickle 

---

### Prerequisites
```bash
pip install torch transformers peft spacy nltk scikit-learn sentence-transformers captum 
python -m spacy download en_core_web_sm
```

---

## Critical Reflections: 

Despite the model's high baseline performance, Task 3's analysis with Task 4 Genetic Algorithm bypass the model pretty easily.The model is fragile not a deep linguistic/sentence pattern analyzer. 

Critical issue that might have caused it : 
### 1. Task 0 Semantic Gap 
*  Could have had better ways to generate AI texts such that semantic similarity is much higher 
(around 0.8). This would've forced the model to learn more about the rhythm and sentence structure rather than taking the short way out. 
This would've single handedly improved the performance by a lot. 

* Ideas:  Use some kind of anchors throughout the generation process. 
    - Include the topics/paragraphs found in original dataset every few thousand words.
    - Now that we have a flow of topics from start to end, you can ask an LLM to fill the gaps in between. 
    - Only include responses which are over 0.8 in similary otherwise reiterate 
 
### 2. Handle overfitting better 
**Adversarial Loop Training**: The bypassed paragraphs generated in Task 4 would be re-injected into the training set labeled as "AI." This closes the "backdoors" we discovered and forces the model to develop a more robust authorship profile.

Moon - PreCog Research Task 2026
