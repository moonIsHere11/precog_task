import os
import numpy as np
import pandas as pd
import spacy
import textstat
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load spacy model
print("Loading spacy model...")
nlp = spacy.load('en_core_web_sm')

def calculate_ttr(text, window_size=150, min_remainder=50):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha]

    n = len(tokens)
    if n < window_size:
        return len(set(tokens)) / n if n > 0 else 0.0

    windows = []
    i = 0

    while i + window_size <= n:
        windows.append(tokens[i:i + window_size])
        i += window_size

    # Handle remainder
    remainder = tokens[i:]
    if len(remainder) >= min_remainder and windows:
        windows[-1].extend(remainder)
    # else: ignore remainder

    # Compute TTR per window
    ttrs = []
    for w in windows:
        if len(w) > 0:
            ttrs.append(len(set(w)) / len(w))

    return float(np.mean(ttrs)) if ttrs else 0.0

def calculate_hapax(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha]
    if len(tokens) == 0:
        return 0.0
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
    return hapax_count / len(tokens)

def calculate_adj_noun_ratio(text):
    doc = nlp(text)
    adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')
    noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
    if noun_count == 0:
        return 0.0
    return adj_count / noun_count

def calculate_dependency_depth(text):
    doc = nlp(text)
    
    def get_depth(token):
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth
    
    if len(list(doc)) == 0:
        return 0.0
    
    depths = [get_depth(token) for token in doc]
    return np.mean(depths) if depths else 0.0

def calculate_punctuation_ratio(text):
    if len(text) == 0:
        return 0.0
    punct_count = sum(1 for char in text if char in '.,;:!?-')
    return punct_count / len(text)

def calculate_flesch_kincaid(text):
    try:
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0.0

def extract_features(text):
    feature_dict = {
        'ttr': calculate_ttr(text),
        'hapax': calculate_hapax(text),
        'adj_noun_ratio': calculate_adj_noun_ratio(text),
        'dependency_depth': calculate_dependency_depth(text),
        'punctuation_ratio': calculate_punctuation_ratio(text),
        'flesch_kincaid': calculate_flesch_kincaid(text)
    }
    # Return as list in the correct order (same as training)
    return [
        feature_dict['ttr'],
        feature_dict['hapax'],
        feature_dict['adj_noun_ratio'],
        feature_dict['dependency_depth'],
        feature_dict['punctuation_ratio'],
        feature_dict['flesch_kincaid']
    ]

def main():
    print("="*80)
    print("XGBoost Model Evaluation on gtest_data.csv")
    print("="*80)
    
    # Define paths
    model_path = '/home/mohit/projects/precog_task/task_2/xgb_text_classifier.json'
    gtest_path = '/home/mohit/projects/precog_task/task_2/gtest_data.csv'
    train_csv_path = '/home/mohit/projects/precog_task/task_2/xgb/xgb_all_train.csv'
    
    # Load the trained model
    print("\n[Step 1] Loading trained XGBoost model...")
    print("-" * 80)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Load training data to fit the scaler
    print("\n[Step 2] Loading training data to fit scaler...")
    print("-" * 80)
    train_df = pd.read_csv(train_csv_path)
    feature_cols = ['ttr', 'hapax', 'adj_noun_ratio', 'dependency_depth', 
                    'punctuation_ratio', 'flesch_kincaid']
    X_train = train_df[feature_cols].values
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"Scaler fitted on {len(X_train)} training samples")
    
    # Load gtest_data.csv
    print("\n[Step 3] Loading gtest_data.csv...")
    print("-" * 80)
    if not os.path.exists(gtest_path):
        print(f"ERROR: {gtest_path} not found!")
        return
    
    gtest_df = pd.read_csv(gtest_path)
    print(f"Loaded {len(gtest_df)} samples")
    print(f"\nLabel distribution:")
    label_counts = gtest_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = ['Human', 'AI', 'AI Mimic'][label]
        print(f"  {label} ({label_name}): {count} samples")
    
    # Extract features
    print("\n[Step 4] Extracting features from gtest samples...")
    print("-" * 80)
    gtest_features = []
    gtest_labels = []
    gtest_texts = []
    failed_count = 0
    
    for idx, row in gtest_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Processing sample {idx+1}/{len(gtest_df)}...")
        try:
            features = extract_features(row['text'])
            gtest_features.append(features)
            gtest_labels.append(row['label'])
            gtest_texts.append(row['text'])
        except Exception as e:
            failed_count += 1
            print(f"  Warning: Failed to extract features for sample {idx}: {e}")
    
    X_gtest = np.array(gtest_features)
    y_gtest = np.array(gtest_labels)
    
    print(f"Successfully extracted features from {len(X_gtest)} samples")
    if failed_count > 0:
        print(f"  Warning: {failed_count} samples failed feature extraction")
    
    # Scale features
    print("\n[Step 5] Scaling features...")
    print("-" * 80)
    X_gtest_scaled = scaler.transform(X_gtest)
    print(f"Features scaled")
    
    # Make predictions
    print("\n[Step 6] Making predictions...")
    print("-" * 80)
    y_gtest_pred = model.predict(X_gtest_scaled)
    print(f"Predictions completed")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION RESULTS ON gtest_data.csv")
    print("="*80)
    
    print(f"\nAccuracy: {accuracy_score(y_gtest, y_gtest_pred):.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_gtest, y_gtest_pred, 
                                target_names=['Human', 'AI', 'AI Mimic']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_gtest, y_gtest_pred)
    print(cm)
    print("\nConfusion Matrix Interpretation:")
    print("             Predicted:")
    print("             Human    AI    AI Mimic")
    print(f"Actual Human:   {cm[0][0]:3d}    {cm[0][1]:3d}      {cm[0][2]:3d}")
    print(f"Actual AI:      {cm[1][0]:3d}    {cm[1][1]:3d}      {cm[1][2]:3d}")
    print(f"Actual AI Mimic:{cm[2][0]:3d}    {cm[2][1]:3d}      {cm[2][2]:3d}")
    
    # Save predictions
    print("\n[Step 7] Saving predictions...")
    print("-" * 80)
    results_df = pd.DataFrame({
        'text': gtest_texts,
        'true_label': y_gtest,
        'predicted_label': y_gtest_pred,
        'true_class': [['Human', 'AI', 'AI Mimic'][label] for label in y_gtest],
        'predicted_class': [['Human', 'AI', 'AI Mimic'][label] for label in y_gtest_pred]
    })
    
    results_path = '/home/mohit/projects/precog_task/task_2/gtest_predictions.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Predictions saved to: {results_path}")
    
    # Show some example misclassifications
    print("\n[Step 8] Analyzing misclassifications...")
    print("-" * 80)
    misclassified = results_df[results_df['true_label'] != results_df['predicted_label']]
    print(f"Total misclassifications: {len(misclassified)}/{len(results_df)} ({len(misclassified)/len(results_df)*100:.1f}%)")
    
    if len(misclassified) > 0:
        print("\nMisclassification breakdown:")
        for true_label in [0, 1, 2]:
            for pred_label in [0, 1, 2]:
                if true_label != pred_label:
                    count = len(misclassified[(misclassified['true_label'] == true_label) & 
                                             (misclassified['predicted_label'] == pred_label)])
                    if count > 0:
                        true_name = ['Human', 'AI', 'AI Mimic'][true_label]
                        pred_name = ['Human', 'AI', 'AI Mimic'][pred_label]
                        print(f"  {true_name} → {pred_name}: {count} samples")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
