"""
XGBoost Classifier for Human vs AI vs AI Mimic Human Text Classification

This script implements a machine learning pipeline to classify text into three categories:
- Class 1: Human-written text
- Class 2: AI-generated text
- Class 3: AI-generated text mimicking human style

Features extracted:
1. TTR (Type-Token Ratio)
2. Hapax Legomena (words appearing only once)
3. Adjective vs Noun ratio
4. Dependency depth (average parse tree depth)
5. Punctuation ratio (semicolons, em-dashes, dashes)
6. Flesch-Kincaid Grade Level
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import spacy
from collections import Counter
import textstat
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# Load spacy model
print("Loading spacy model...")
nlp = spacy.load('en_core_web_sm')


def split_into_paragraphs(text, min_words=100, max_words=200):
    # Split text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    paragraphs = []
    current_paragraph = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words and we already have min_words
        if current_word_count + sentence_words > max_words and current_word_count >= min_words:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentence]
            current_word_count = sentence_words
        else:
            current_paragraph.append(sentence)
            current_word_count += sentence_words
    
    # Add remaining paragraph if it meets minimum requirement
    if current_word_count >= min_words:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs


def load_data_from_folder(folder_path, max_paragraphs_per_file=None):
    all_paragraphs = []
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    print(f"Processing {len(files)} files from {folder_path}")
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        paragraphs = split_into_paragraphs(text)
        
        if max_paragraphs_per_file:
            paragraphs = paragraphs[:max_paragraphs_per_file]
        
        all_paragraphs.extend(paragraphs)
        print(f"  {os.path.basename(file_path)}: {len(paragraphs)} paragraphs")
    
    print(f"Total paragraphs from {folder_path}: {len(all_paragraphs)}")
    return all_paragraphs


def calculate_ttr(text):
    """Calculate Type-Token Ratio (TTR)"""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    unique_words = len(set(words))
    return unique_words / len(words)


def calculate_hapax(text):
    """Calculate Hapax Legomena ratio (words appearing only once)"""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    word_counts = Counter(words)
    hapax_count = sum(1 for count in word_counts.values() if count == 1)
    return hapax_count / len(words)


def calculate_adj_noun_ratio(text):
    """Calculate Adjective to Noun ratio using spacy"""
    doc = nlp(text)
    adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')
    noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
    
    if noun_count == 0:
        return 0.0
    return adj_count / noun_count


def calculate_dependency_depth(text):
    """Calculate average dependency tree depth using spacy"""
    doc = nlp(text)
    
    def get_depth(token):
        """Recursively calculate depth of a token in the dependency tree"""
        if token.head == token:  # Root token
            return 0
        return 1 + get_depth(token.head)
    
    depths = [get_depth(token) for token in doc]
    
    if len(depths) == 0:
        return 0.0
    return np.mean(depths)


def calculate_punctuation_ratio(text):
    """Calculate ratio of semicolons, em-dashes, and dashes to total text length"""
    # Count semicolons, em-dashes (—), en-dashes (–), and hyphens (-)
    special_punct = re.findall(r'[;—–-]', text)
    
    if len(text) == 0:
        return 0.0
    return len(special_punct) / len(text)


def calculate_flesch_kincaid(text):
    """Calculate Flesch-Kincaid Grade Level"""
    try:
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0.0


def extract_features(text):
    """
    Extract all features from a text paragraph.
    
    Returns:
        Dictionary of features
    """
    features = {
        'ttr': calculate_ttr(text),
        'hapax': calculate_hapax(text),
        'adj_noun_ratio': calculate_adj_noun_ratio(text),
        'dependency_depth': calculate_dependency_depth(text),
        'punctuation_ratio': calculate_punctuation_ratio(text),
        'flesch_kincaid': calculate_flesch_kincaid(text)
    }
    return features



def main():
    print("="*80)
    print("XGBoost Text Classification Pipeline")
    print("="*80)
    
    # Define paths
    base_path = '/home/mohit/projects/precog_task/datasets'
    class_1_path = os.path.join(base_path, 'class_1')
    class_2_path = os.path.join(base_path, 'class_2')
    class_3_path = os.path.join(base_path, 'class_3')
    
    print("\n[Step 1] Loading data from datasets...")
    print("-" * 80)
    
    # Class 1: Human text - load ALL paragraphs first, then split
    class_1_all_paragraphs = load_data_from_folder(class_1_path, max_paragraphs_per_file=None)
    
    # Class 2 and 3: AI text - use all paragraphs
    class_2_paragraphs = load_data_from_folder(class_2_path, max_paragraphs_per_file=None)
    class_3_paragraphs = load_data_from_folder(class_3_path, max_paragraphs_per_file=None)
    
    # Step 2: Split ALL classes into train/val/test
    print("\n[Step 2] Splitting all classes into train/val/test...")
    print("-" * 80)
    
    # Set sizes
    test_size = 50
    val_size = 50
    
    np.random.seed(42)
    
    # Split class 1 (Human): 150 for train, 50 for val, 50 for test (per file equivalent)
    # We have 4 files, so we should use 150*4=600 for train, and remaining for val/test
    if len(class_1_all_paragraphs) < 700:  # Need at least 600 train + 50 val + 50 test
        print(f"Warning: Class 1 has only {len(class_1_all_paragraphs)} paragraphs.")
        # Use proportional split
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        n_train_1 = int(len(class_1_all_paragraphs) * train_ratio)
        n_val_1 = int(len(class_1_all_paragraphs) * val_ratio)
        n_test_1 = len(class_1_all_paragraphs) - n_train_1 - n_val_1
    else:
        # Use first 600 for train, next 50 for val, next 50 for test
        n_train_1 = 600
        n_val_1 = 50
        n_test_1 = 50
    
    indices_1 = np.random.permutation(len(class_1_all_paragraphs))
    class_1_train_idx = indices_1[:n_train_1]
    class_1_val_idx = indices_1[n_train_1:n_train_1 + n_val_1]
    class_1_test_idx = indices_1[n_train_1 + n_val_1:n_train_1 + n_val_1 + n_test_1]
    
    class_1_train = [class_1_all_paragraphs[i] for i in class_1_train_idx]
    class_1_val = [class_1_all_paragraphs[i] for i in class_1_val_idx]
    class_1_test = [class_1_all_paragraphs[i] for i in class_1_test_idx]

    
    # Split class 2 (AI)
    if len(class_2_paragraphs) < test_size + val_size:
        print(f"Warning: Class 2 has only {len(class_2_paragraphs)} paragraphs. Adjusting split sizes.")
        test_size_2 = min(test_size, len(class_2_paragraphs) // 3)
        val_size_2 = min(val_size, len(class_2_paragraphs) // 3)
    else:
        test_size_2 = test_size
        val_size_2 = val_size
    
    indices_2 = np.random.permutation(len(class_2_paragraphs))
    class_2_test_idx = indices_2[:test_size_2]
    class_2_val_idx = indices_2[test_size_2:test_size_2 + val_size_2]
    class_2_train_idx = indices_2[test_size_2 + val_size_2:]
    
    class_2_test = [class_2_paragraphs[i] for i in class_2_test_idx]
    class_2_val = [class_2_paragraphs[i] for i in class_2_val_idx]
    class_2_train = [class_2_paragraphs[i] for i in class_2_train_idx]
    
    # Split class 3 (AI Mimic)
    if len(class_3_paragraphs) < test_size + val_size:
        print(f"Warning: Class 3 has only {len(class_3_paragraphs)} paragraphs. Adjusting split sizes.")
        test_size_3 = min(test_size, len(class_3_paragraphs) // 3)
        val_size_3 = min(val_size, len(class_3_paragraphs) // 3)
    else:
        test_size_3 = test_size
        val_size_3 = val_size
    

    indices_3 = np.random.permutation(len(class_3_paragraphs))
    class_3_test_idx = indices_3[:test_size_3]
    class_3_val_idx = indices_3[test_size_3:test_size_3 + val_size_3]
    class_3_train_idx = indices_3[test_size_3 + val_size_3:]
    
    class_3_test = [class_3_paragraphs[i] for i in class_3_test_idx]
    class_3_val = [class_3_paragraphs[i] for i in class_3_val_idx]
    class_3_train = [class_3_paragraphs[i] for i in class_3_train_idx]
    
    print(f"Class 1 (Human) - Train: {len(class_1_train)}, Val: {len(class_1_val)}, Test: {len(class_1_test)} paragraphs")
    print(f"Class 2 (AI) - Train: {len(class_2_train)}, Val: {len(class_2_val)}, Test: {len(class_2_test)}")
    print(f"Class 3 (AI Mimic) - Train: {len(class_3_train)}, Val: {len(class_3_val)}, Test: {len(class_3_test)}")
    
    # Step 3: Extract features
    print("\n[Step 3] Extracting features...")
    print("-" * 80)
    

    def extract_features_for_dataset(paragraphs, label):
        """Extract features for a list of paragraphs with progress reporting"""
        features_list = []
        for i, para in enumerate(paragraphs):
            if (i + 1) % 50 == 0:
                print(f"  Processing paragraph {i+1}/{len(paragraphs)}...")
            features = extract_features(para)
            features['label'] = label
            features['text'] = para[:100] + '...'  # Store first 100 chars for reference
            features_list.append(features)
        return features_list
    
    print("Extracting features for Class 1 (Human) - Train...")
    train_features_1 = extract_features_for_dataset(class_1_train, 0)
    
    print("Extracting features for Class 1 (Human) - Val...")
    val_features_1 = extract_features_for_dataset(class_1_val, 0)
    
    print("Extracting features for Class 1 (Human) - Test...")
    test_features_1 = extract_features_for_dataset(class_1_test, 0)
    
    print("Extracting features for Class 2 (AI) - Train...")
    train_features_2 = extract_features_for_dataset(class_2_train, 1)
    
    print("Extracting features for Class 2 (AI) - Val...")
    val_features_2 = extract_features_for_dataset(class_2_val, 1)
    
    print("Extracting features for Class 2 (AI) - Test...")
    test_features_2 = extract_features_for_dataset(class_2_test, 1)
    
    print("Extracting features for Class 3 (AI Mimic) - Train...")
    train_features_3 = extract_features_for_dataset(class_3_train, 2)
    
    print("Extracting features for Class 3 (AI Mimic) - Val...")
    val_features_3 = extract_features_for_dataset(class_3_val, 2)
    
    print("Extracting features for Class 3 (AI Mimic) - Test...")
    test_features_3 = extract_features_for_dataset(class_3_test, 2)
    
    # Step 4: Create DataFrames and save to CSV
    print("\n[Step 4] Creating DataFrames and saving to CSV...")
    print("-" * 80)
    
    # Training data: all class 1 train + class 2 train + class 3 train
    train_df = pd.DataFrame(train_features_1 + train_features_2 + train_features_3)
    
    # Validation data: class 1 val + class 2 val + class 3 val
    val_df = pd.DataFrame(val_features_1 + val_features_2 + val_features_3)
    
    # Test data: class 1 test + class 2 test + class 3 test
    test_df = pd.DataFrame(test_features_1 + test_features_2 + test_features_3)
    
    # Save feature CSVs
    output_dir = '/home/mohit/projects/precog_task/task_2'
    
    # Save individual class files for reference
    train_df_1 = pd.DataFrame(train_features_1)
    val_df_1 = pd.DataFrame(val_features_1)
    test_df_1 = pd.DataFrame(test_features_1)
    train_df_2 = pd.DataFrame(train_features_2)
    train_df_3 = pd.DataFrame(train_features_3)
    val_df_2 = pd.DataFrame(val_features_2)
    val_df_3 = pd.DataFrame(val_features_3)
    test_df_2 = pd.DataFrame(test_features_2)
    test_df_3 = pd.DataFrame(test_features_3)
    
    train_df_1.to_csv(os.path.join(output_dir, 'xgb_class1_train.csv'), index=False)
    val_df_1.to_csv(os.path.join(output_dir, 'xgb_class1_val.csv'), index=False)
    test_df_1.to_csv(os.path.join(output_dir, 'xgb_class1_test.csv'), index=False)
    train_df_2.to_csv(os.path.join(output_dir, 'xgb_class2_train.csv'), index=False)
    val_df_2.to_csv(os.path.join(output_dir, 'xgb_class2_val.csv'), index=False)
    test_df_2.to_csv(os.path.join(output_dir, 'xgb_class2_test.csv'), index=False)
    train_df_3.to_csv(os.path.join(output_dir, 'xgb_class3_train.csv'), index=False)
    val_df_3.to_csv(os.path.join(output_dir, 'xgb_class3_val.csv'), index=False)
    test_df_3.to_csv(os.path.join(output_dir, 'xgb_class3_test.csv'), index=False)
    
    # Save combined datasets
    train_df.to_csv(os.path.join(output_dir, 'xgb_all_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'xgb_all_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'xgb_all_test.csv'), index=False)
    
    print(f"Saved feature CSVs to {output_dir}")
    print(f"  - Training data: {len(train_df)} samples")
    print(f"  - Validation data: {len(val_df)} samples")
    print(f"  - Test data: {len(test_df)} samples")
    
    # Step 5: Hyperparameter Tuning with Regularization
    print("\n[Step 5] Hyperparameter Tuning (RandomizedSearchCV + Regularization)...")
    print("-" * 80)
    
    # Prepare feature columns
    feature_cols = ['ttr', 'hapax', 'adj_noun_ratio', 'dependency_depth', 
                    'punctuation_ratio', 'flesch_kincaid']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Combine train + val for cross-validated search, then retrain best on train only
    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])
    
    # --- Hyperparameter search space (includes regularization) ---
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),       # [0.01, 0.30]
        'subsample': uniform(0.6, 0.4),              # [0.6, 1.0]
        'colsample_bytree': uniform(0.5, 0.5),       # [0.5, 1.0]
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 5),                       # min split loss
        'reg_alpha': uniform(0, 2),                   # L1 regularization
        'reg_lambda': uniform(0.5, 4.5),              # L2 regularization [0.5, 5.0]
    }
    
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=60,                # number of random combos to try
        scoring='accuracy',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        refit=True,               # refit best estimator on full trainval
    )
    
    print("Running RandomizedSearchCV (60 iterations, 5-fold CV)...")
    search.fit(X_trainval, y_trainval)
    
    print(f"\nBest CV Accuracy: {search.best_score_:.4f}")
    print(f"Best Parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # --- Retrain best params on train-only with early stopping against val ---
    print("\nRetraining best params on training set with early stopping...")
    best_params = search.best_params_.copy()
    
    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        early_stopping_rounds=20,
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    # Step 6: Evaluate model
    print("\n[Step 6] Evaluating model...")
    print("-" * 80)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Training set performance
    print("\n--- Training Set Performance ---")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_train_pred, 
                                target_names=['Human', 'AI', 'AI Mimic']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    
    # Validation set performance
    print("\n--- Validation Set Performance ---")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, 
                                target_names=['Human', 'AI', 'AI Mimic']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    # Test set performance
    print("\n--- Test Set Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Human', 'AI', 'AI Mimic']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Feature importance
    print("\n--- Feature Importance ---")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Save the model
    model_path = os.path.join(output_dir, 'xgb_text_classifier.json')
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Test on gtest_data.csv if it exists
    gtest_path = os.path.join(output_dir, 'gtest_data.csv')
    if os.path.exists(gtest_path):
        print("\n" + "="*80)
        print("[Additional Test] Evaluating on gtest_data.csv")
        print("="*80)
        
        # Load gtest_data
        gtest_df = pd.read_csv(gtest_path)
        print(f"\nLoaded gtest_data.csv: {len(gtest_df)} samples")
        print(f"Label distribution:\n{gtest_df['label'].value_counts().sort_index()}")
        
        # Extract features
        print("\nExtracting features from gtest samples...")
        gtest_features = []
        gtest_labels = []
        gtest_texts = []
        
        for idx, row in gtest_df.iterrows():
            features = extract_features(row['text'])
            if features is not None:
                gtest_features.append(features)
                gtest_labels.append(row['label'])
                gtest_texts.append(row['text'])
        
        X_gtest = np.array(gtest_features)
        y_gtest = np.array(gtest_labels)
        
        print(f"Successfully extracted features from {len(X_gtest)} samples")
        
        # Scale features using the same scaler
        X_gtest_scaled = scaler.transform(X_gtest)
        
        # Predict
        y_gtest_pred = model.predict(X_gtest_scaled)
        
        # Evaluate
        print("\n--- gtest_data.csv Performance ---")
        print(f"Accuracy: {accuracy_score(y_gtest, y_gtest_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_gtest, y_gtest_pred, 
                                    target_names=['Human', 'AI', 'AI Mimic']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_gtest, y_gtest_pred))
        
        # Save predictions
        gtest_results = pd.DataFrame({
            'text': gtest_texts,
            'true_label': y_gtest,
            'predicted_label': y_gtest_pred
        })
        gtest_results_path = os.path.join(output_dir, 'gtest_predictions.csv')
        gtest_results.to_csv(gtest_results_path, index=False)
        print(f"\nPredictions saved to: {gtest_results_path}")
    else:
        print(f"\nNote: gtest_data.csv not found at {gtest_path}, skipping additional test.")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
