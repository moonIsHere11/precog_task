import os
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle

print("Loading spacy model...")
nlp = spacy.load('en_core_web_sm')

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextDataset(Dataset):
    """PyTorch Dataset for text classification"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class FeedForwardNN(nn.Module):
    """Feedforward Neural Network for binary classification"""
    def __init__(self, input_dim=300):
        super(FeedForwardNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)  # No sigmoid here
        )
    
    def forward(self, x):
        return self.network(x)


def text_to_embedding(text, fasttext_model, embedding_dim=300):
    """Convert text to FastText embedding by averaging word vectors"""
    doc = nlp(text.lower())
    vectors = []
    
    for token in doc:
        if token.is_alpha:
            try:
                vectors.append(fasttext_model[token.text])
            except KeyError:
                # Handle OOV words - use zero vector
                pass
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    
    # Average and normalize
    embedding = np.mean(vectors, axis=0)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def evaluate_model(model, data_loader):
    """Evaluate model and return predictions"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Get raw logits
            outputs = model(embeddings).squeeze()
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probs)


def main():
    print("="*80)
    print("Neural Network Evaluation on gtest_data.csv")
    print("="*80)
    
    # Define paths
    model_path = '/home/mohit/projects/precog_task/task_2/tier_B/nn_model.pth'
    gtest_path = '/home/mohit/projects/precog_task/datasets/gtest_data.csv'
    output_dir = '/home/mohit/projects/precog_task/task_2/tier_B'
    
    # Step 1: Load the trained model
    print("\n[Step 1] Loading trained Neural Network model...")
    print("-" * 80)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first by running tier_b.py")
        return
    
    model = FeedForwardNN(input_dim=300).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from: {model_path}")
    
    # Step 2: Load FastText model
    print("\n[Step 2] Loading FastText model...")
    print("-" * 80)
    print("Loading cached model...")
    fasttext_model = api.load('fasttext-wiki-news-subwords-300')
    embedding_dim = 300
    print(f"✓ FastText model loaded (dimension: {embedding_dim})")
    
    # Step 3: Load gtest_data.csv
    print("\n[Step 3] Loading gtest_data.csv...")
    print("-" * 80)
    
    if not os.path.exists(gtest_path):
        print(f"ERROR: {gtest_path} not found!")
        return
    
    gtest_df = pd.read_csv(gtest_path)
    print(f"✓ Loaded {len(gtest_df)} samples")
    
    # Convert labels: 0=Human (keep as 0), 1=AI (keep as 1), 2=AI-Mimic (convert to 1)
    # Since this is binary classification (Human vs AI), treat AI-Mimic as AI
    gtest_df['binary_label'] = gtest_df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    print(f"\nOriginal label distribution:")
    for label, count in gtest_df['label'].value_counts().sort_index().items():
        label_name = ['Human', 'AI', 'AI Mimic'][label]
        print(f"  {label} ({label_name}): {count} samples")
    
    print(f"\nBinary label distribution:")
    for label, count in gtest_df['binary_label'].value_counts().sort_index().items():
        label_name = ['Human', 'AI'][label]
        print(f"  {label} ({label_name}): {count} samples")
    
    # Step 4: Generate embeddings
    print("\n[Step 4] Generating embeddings for gtest samples...")
    print("-" * 80)
    
    gtest_embeddings = []
    gtest_labels = []
    gtest_texts = []
    gtest_original_labels = []
    failed_count = 0
    
    for idx, row in gtest_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Processing sample {idx+1}/{len(gtest_df)}...")
        try:
            emb = text_to_embedding(row['text'], fasttext_model, embedding_dim)
            gtest_embeddings.append(emb)
            gtest_labels.append(row['binary_label'])
            gtest_original_labels.append(row['label'])
            gtest_texts.append(row['text'])
        except Exception as e:
            failed_count += 1
            print(f"  Warning: Failed to process sample {idx}: {e}")
    
    gtest_embeddings = np.array(gtest_embeddings)
    gtest_labels = np.array(gtest_labels)
    gtest_original_labels = np.array(gtest_original_labels)
    
    print(f"✓ Successfully generated embeddings for {len(gtest_embeddings)} samples")
    if failed_count > 0:
        print(f"  Warning: {failed_count} samples failed")
    
    # Step 5: Create DataLoader
    print("\n[Step 5] Creating DataLoader...")
    print("-" * 80)
    
    gtest_dataset = TextDataset(gtest_embeddings, gtest_labels)
    gtest_loader = DataLoader(gtest_dataset, batch_size=32, shuffle=False)
    print(f"✓ DataLoader created")
    
    # Step 6: Evaluate
    print("\n[Step 6] Making predictions...")
    print("-" * 80)
    
    test_labels, test_predictions, test_probs = evaluate_model(model, gtest_loader)
    print(f"✓ Predictions completed")
    
    # Step 7: Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS ON gtest_data.csv")
    print("="*80)
    
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, zero_division=0)
    recall = recall_score(test_labels, test_predictions, zero_division=0)
    f1 = f1_score(test_labels, test_predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(test_labels, test_probs)
        print(f"\nAUC-ROC: {auc:.4f}")
    except:
        auc = None
        print(f"\nAUC-ROC: Could not calculate (likely only one class present)")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(test_labels, test_predictions, 
                                target_names=['Human', 'AI'], 
                                digits=4,
                                zero_division=0))
    
    print("\n--- Confusion Matrix (Binary) ---")
    cm = confusion_matrix(test_labels, test_predictions)
    print(cm)
    print("\nConfusion Matrix Interpretation:")
    print("             Predicted:")
    print("             Human    AI")
    print(f"Actual Human:   {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"Actual AI:      {cm[1][0]:3d}    {cm[1][1]:3d}")
    
    # Breakdown by original 3-class labels
    print("\n--- Performance Breakdown by Original Classes ---")
    for orig_label in [0, 1, 2]:
        mask = (gtest_original_labels == orig_label)
        if mask.sum() > 0:
            label_name = ['Human', 'AI', 'AI Mimic'][orig_label]
            label_acc = accuracy_score(test_labels[mask], test_predictions[mask])
            label_f1 = f1_score(test_labels[mask], test_predictions[mask], zero_division=0)
            print(f"{label_name}: Accuracy={label_acc:.4f}, F1={label_f1:.4f} ({mask.sum()} samples)")
    
    # Step 8: Save results
    print("\n[Step 8] Saving results...")
    print("-" * 80)
    
    # Save predictions
    results_df = pd.DataFrame({
        'text': gtest_texts,
        'original_label': gtest_original_labels,
        'original_class': [['Human', 'AI', 'AI Mimic'][label] for label in gtest_original_labels],
        'binary_true_label': test_labels,
        'binary_predicted_label': test_predictions,
        'probability_ai': test_probs,
        'true_class': [['Human', 'AI'][int(label)] for label in test_labels],
        'predicted_class': [['Human', 'AI'][int(label)] for label in test_predictions]
    })
    
    results_path = os.path.join(output_dir, 'gtest_nn_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Predictions saved to: {results_path}")
    
    # Step 9: Plot ROC curve if possible
    if auc is not None:
        print("\n[Step 9] Saving ROC curve...")
        print("-" * 80)
        
        fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - gtest_data.csv', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(output_dir, 'gtest_roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {roc_path}")
    
    # Analyze misclassifications
    print("\n[Step 10] Analyzing misclassifications...")
    print("-" * 80)
    
    misclassified = results_df[results_df['binary_true_label'] != results_df['binary_predicted_label']]
    print(f"Total misclassifications: {len(misclassified)}/{len(results_df)} ({len(misclassified)/len(results_df)*100:.1f}%)")
    
    if len(misclassified) > 0:
        print("\nMisclassification breakdown (binary):")
        for true_label in [0, 1]:
            for pred_label in [0, 1]:
                if true_label != pred_label:
                    count = len(misclassified[(misclassified['binary_true_label'] == true_label) & 
                                             (misclassified['binary_predicted_label'] == pred_label)])
                    if count > 0:
                        true_name = ['Human', 'AI'][true_label]
                        pred_name = ['Human', 'AI'][pred_label]
                        print(f"  {true_name} → {pred_name}: {count} samples")
        
        print("\nMisclassification breakdown (by original 3-class):")
        for orig_label in [0, 1, 2]:
            orig_name = ['Human', 'AI', 'AI Mimic'][orig_label]
            orig_misclassified = misclassified[misclassified['original_label'] == orig_label]
            if len(orig_misclassified) > 0:
                print(f"  {orig_name}: {len(orig_misclassified)} misclassified")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)
    
    print("\nSummary:")
    print(f"  Model: {model_path}")
    print(f"  Test samples: {len(results_df)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Predictions saved: {results_path}")


if __name__ == '__main__':
    main()
