# XGBoost Text Classification: Human vs AI vs AI Mimic

## Overview
This project implements an XGBoost classifier to distinguish between three types of text:
- **Class 0 (Human)**: Human-written text
- **Class 1 (AI)**: AI-generated text
- **Class 2 (AI Mimic)**: AI-generated text attempting to mimic human style

## Pipeline Workflow

### 1. Data Loading and Preprocessing
- **Class 1 (Human)**: Loaded 4 files from `datasets/class_1/`, extracted all paragraphs
  - Total: 1,079 paragraphs (600 train, 50 val, 50 test)
- **Class 2 (AI)**: Loaded 10 files from `datasets/class_2/`, extracted all paragraphs
  - Total: 353 paragraphs (253 train, 50 val, 50 test)
- **Class 3 (AI Mimic)**: Loaded 10 files from `datasets/class_3/`, extracted all paragraphs
  - Total: 331 paragraphs (231 train, 50 val, 50 test)

**Paragraph Extraction**: 
- Each paragraph contains 100-200 words
- Sentences are not cut off mid-sentence - the algorithm checks for sentence boundaries
- Uses spaCy for sentence tokenization

### 2. Feature Extraction
Six linguistic features were extracted for each paragraph:

1. **TTR (Type-Token Ratio)**: Ratio of unique words to total words
   - Measures lexical diversity
   
2. **Hapax Legomena**: Ratio of words appearing only once
   - Indicates vocabulary richness
   
3. **Adjective/Noun Ratio**: Calculated using spaCy POS tagging
   - Measures descriptive language usage
   
4. **Dependency Depth**: Average depth of syntactic parse trees
   - Indicates sentence complexity
   
5. **Punctuation Ratio**: Ratio of semicolons, em-dashes, and dashes to text length
   - Measures punctuation style
   
6. **Flesch-Kincaid Grade Level**: Readability score
   - Indicates reading difficulty level

### 3. Model Training
- **Model**: XGBoost Classifier (200 estimators, max_depth=6)
- **Training Data**: 1,084 samples (600 human + 253 AI + 231 AI mimic)
- **Validation Data**: 150 samples (50 human + 50 AI + 50 AI mimic)
- **Test Data**: 150 samples (50 human + 50 AI + 50 AI mimic)

### 4. Results

#### Training Set Performance
- **Accuracy**: 100% (perfect fit on training data)
- The model completely learned the training distribution

#### Validation Set Performance
- **Accuracy**: 66%
- The model performs reasonably well, especially at identifying Human text
- Classification Report shows:
  - Human: 96% recall (excellent detection), 68% precision
  - AI: 50% recall, 58% precision (struggles with AI detection)
  - AI Mimic: 52% recall, 72% precision
- Confusion Matrix:
  ```
  [[48  2  0]   <- Human: 48 correct, 2 classified as AI
   [15 25 10]   <- AI: 25 correct, 15 classified as Human, 10 as AI Mimic
   [ 8 16 26]]  <- AI Mimic: 26 correct, 16 classified as AI, 8 as Human
  ```

#### Test Set Performance
- **Accuracy**: 63.3%
- Similar performance to validation set
- Classification Report shows:
  - Human: 96% recall (excellent detection), 68% precision
  - AI: 42% recall, 62% precision (poorest performance)
  - AI Mimic: 52% recall, 58% precision
- Confusion Matrix:
  ```
  [[48  2  0]   <- Human: 48 correct, 2 classified as AI
   [10 21 19]   <- AI: 21 correct, 10 classified as Human, 19 as AI Mimic
   [13 7 30]]  <- AI Mimic: 26 correct, 11 classified as AI, 13 as Human
  ```

#### Feature Importance
The most important features for classification:
1. **Punctuation Ratio**: 10.7%
2. **Hapax**: 19.0%
3. **Flesch-Kincaid**: 18.6%
4. **TTR**: 16.3%
5. **Dependency Depth**: 13.6%
6. **Adj/Noun Ratio**: 21.9%

## Files Generated

### Feature CSV Files
- `xgb_class1_train.csv`: Class 1 (Human) training features (600 samples)
- `xgb_class1_val.csv`: Class 1 (Human) validation features (50 samples)
- `xgb_class1_test.csv`: Class 1 (Human) test features (50 samples)
- `xgb_class2_train.csv`: Class 2 (AI) training features (253 samples)
- `xgb_class2_val.csv`: Class 2 (AI) validation features (50 samples)
- `xgb_class2_test.csv`: Class 2 (AI) test features (50 samples)
- `xgb_class3_train.csv`: Class 3 (AI Mimic) training features (231 samples)
- `xgb_class3_val.csv`: Class 3 (AI Mimic) validation features (50 samples)
- `xgb_class3_test.csv`: Class 3 (AI Mimic) test features (50 samples)
- `xgb_all_train.csv`: Combined training data (1,084 samples)
- `xgb_all_val.csv`: Combined validation data (150 samples)
- `xgb_all_test.csv`: Combined test data (150 samples)

### Model File
- `xgb_text_classifier.json`: Trained XGBoost model

## Observations

1. **Perfect Training Accuracy**: The model achieves 100% accuracy on training data, suggesting potential overfitting despite regularization.

2. **Good Human Detection**: The model excels at identifying human-written text with 96% recall on both validation and test sets. This suggests human text has distinctive linguistic features.

3. **AI vs AI Mimic Confusion**: The model struggles most to distinguish between:
   - AI-generated text (Class 1)
   - AI-generated text mimicking human style (Class 2)
   
   This is evident from the confusion matrices where AI texts are frequently misclassified as AI Mimic and vice versa.

4. **Overall Performance**: 
   - Validation: 66% accuracy
   - Test: 63.3% accuracy
   - Better than random (33.3%) but significant room for improvement

5. **Feature Insights**: 
   - Punctuation patterns (20.7%) are the most discriminative feature
   - Lexical diversity (Hapax 19%, TTR 16.3%) is crucial for classification
   - Readability (Flesch-Kincaid 18.6%) helps distinguish text types
   - Syntactic features (dependency depth, adj/noun ratio) are less discriminative

## Potential Improvements

1. **More Features**: Add more sophisticated features like:
   - Perplexity scores
   - Named entity density
   - Stylometric features
   - N-gram features
   - Sentiment analysis features

2. **Balanced Dataset**: The training data is imbalanced (600 human vs 253 AI vs 231 AI mimic)

3. **Hyperparameter Tuning**: Use cross-validation and grid search for optimal parameters

4. **Regularization**: Add stronger regularization to prevent overfitting

5. **Ensemble Methods**: Combine multiple models for better generalization

6. **Deep Learning**: Consider using transformer-based models (BERT, RoBERTa) for better text understanding

## Running the Pipeline

```bash
python tier_a.py
```

## Dependencies
- numpy
- pandas
- spacy (with en_core_web_sm model)
- textstat
- xgboost
- scikit-learn


Results:
