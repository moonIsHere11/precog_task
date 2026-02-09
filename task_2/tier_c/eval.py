import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, precision_score, 
                             recall_score)
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextDataset(Dataset):
    """Dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, data_loader):
    """Evaluate model and return predictions"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities using softmax
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of AI class
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probs)


def main():
    print("="*80)
    print("DistilBERT + LoRA Evaluation on gtest_data.csv")
    print("="*80)
    
    # Define paths
    base_model_name = 'distilbert-base-uncased'
    lora_model_path = '/home/mohit/projects/precog_task/task_2/tier_c/distilbert_lora_model'
    gtest_path = '/home/mohit/projects/precog_task/datasets/gtest_data.csv'
    output_dir = '/home/mohit/projects/precog_task/task_2/tier_c'
    
    # Step 1: Load tokenizer
    print("\n[Step 1] Loading tokenizer...")
    print("-" * 80)
    tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
    print(f"✓ Tokenizer loaded")
    
    # Step 2: Load base model and LoRA weights
    print("\n[Step 2] Loading fine-tuned model...")
    print("-" * 80)
    
    # Load base model
    base_model = DistilBertForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {lora_model_path}")
    
    # Step 3: Load gtest_data.csv
    print("\n[Step 3] Loading gtest_data.csv...")
    print("-" * 80)
    
    if not os.path.exists(gtest_path):
        print(f"ERROR: {gtest_path} not found!")
        return
    
    gtest_df = pd.read_csv(gtest_path)
    print(f"✓ Loaded {len(gtest_df)} samples")
    
    # Store original 3-class labels for analysis
    original_labels = gtest_df['label'].values
    
    print(f"\nOriginal label distribution:")
    label_counts = gtest_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = ['Human', 'AI', 'AI Mimic'][label]
        print(f"  {label} ({label_name}): {count} samples")
    
    # Convert to binary labels: 0 = Human, 1 = AI (includes AI Mimic)
    binary_labels = (gtest_df['label'] > 0).astype(int).values
    
    print(f"\nBinary label distribution:")
    print(f"  0 (Human): {(binary_labels == 0).sum()} samples")
    print(f"  1 (AI): {(binary_labels == 1).sum()} samples")
    
    texts = gtest_df['text'].tolist()
    
    # Step 4: Create DataLoader
    print("\n[Step 4] Creating DataLoader...")
    print("-" * 80)
    
    gtest_dataset = TextDataset(texts, binary_labels, tokenizer, max_length=512)
    gtest_loader = DataLoader(gtest_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ DataLoader created with {len(gtest_dataset)} samples")
    
    # Step 5: Make predictions
    print("\n[Step 5] Making predictions...")
    print("-" * 80)
    
    test_labels, test_predictions, test_probs = evaluate_model(model, gtest_loader)
    
    print(f"✓ Predictions completed")
    
    # Step 6: Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS ON gtest_data.csv")
    print("="*80)
    
    test_acc = accuracy_score(test_labels, test_predictions)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions)
    test_recall = recall_score(test_labels, test_predictions)
    
    print(f"\nAUC-ROC: {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(test_labels, test_predictions, 
                                target_names=['Human', 'AI'], digits=4))
    
    print("\n--- Confusion Matrix (Binary) ---")
    cm = confusion_matrix(test_labels, test_predictions)
    print(cm)
    print("\nConfusion Matrix Interpretation:")
    print("             Predicted:")
    print("             Human    AI")
    print(f"Actual Human:   {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"Actual AI:      {cm[1][0]:3d}    {cm[1][1]:3d}")
    
    # Step 7: Analyze performance by original 3-class labels
    print("\n--- Performance Breakdown by Original Classes ---")
    for orig_label in [0, 1, 2]:
        mask = original_labels == orig_label
        if mask.sum() > 0:
            label_name = ['Human', 'AI', 'AI Mimic'][orig_label]
            label_acc = accuracy_score(test_labels[mask], test_predictions[mask])
            label_f1 = f1_score(test_labels[mask], test_predictions[mask], zero_division=0)
            print(f"{label_name}: Accuracy={label_acc:.4f}, F1={label_f1:.4f} ({mask.sum()} samples)")
    
    # Step 8: Save predictions
    print("\n[Step 8] Saving results...")
    print("-" * 80)
    
    results_df = pd.DataFrame({
        'text': texts,
        'original_label': original_labels,
        'binary_label': test_labels,
        'predicted_label': test_predictions,
        'probability_ai': test_probs,
        'original_class': [['Human', 'AI', 'AI Mimic'][label] for label in original_labels],
        'predicted_class': [['Human', 'AI'][label] for label in test_predictions]
    })
    
    results_path = os.path.join(output_dir, 'gtest_distilbert_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f" Predictions saved to: {results_path}")
    
    # Step 9: Save ROC curve
    print("\n[Step 9] Saving ROC curve...")
    print("-" * 80)
    
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - DistilBERT + LoRA on gtest_data.csv', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_dir, 'gtest_distilbert_roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {roc_path}")
    plt.close()
    
    # Step 10: Analyze misclassifications
    print("\n[Step 10] Analyzing misclassifications...")
    print("-" * 80)
    
    misclassified = results_df[results_df['binary_label'] != results_df['predicted_label']]
    print(f"Total misclassifications: {len(misclassified)}/{len(results_df)} ({len(misclassified)/len(results_df)*100:.1f}%)")
    
    if len(misclassified) > 0:
        print("\nMisclassification breakdown (binary):")
        for true_label in [0, 1]:
            for pred_label in [0, 1]:
                if true_label != pred_label:
                    count = len(misclassified[(misclassified['binary_label'] == true_label) & 
                                             (misclassified['predicted_label'] == pred_label)])
                    if count > 0:
                        true_name = ['Human', 'AI'][true_label]
                        pred_name = ['Human', 'AI'][pred_label]
                        print(f"  {true_name} → {pred_name}: {count} samples")
        
        print("\nMisclassification breakdown (by original 3-class):")
        for orig_label in [0, 1, 2]:
            misclass_orig = misclassified[misclassified['original_label'] == orig_label]
            if len(misclass_orig) > 0:
                label_name = ['Human', 'AI', 'AI Mimic'][orig_label]
                print(f"  {label_name}: {len(misclass_orig)} misclassified")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  Model: {lora_model_path}")
    print(f"  Test samples: {len(gtest_df)}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  Predictions saved: {results_path}")


if __name__ == '__main__':
    main()
