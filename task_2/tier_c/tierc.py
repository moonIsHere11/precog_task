import os
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

print("Loading spacy model...")
nlp = spacy.load('en_core_web_sm')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.set_num_threads(4)

def split_into_paragraphs(text, min_words=150, max_words=300):
    """Split text into paragraphs of 150-300 words"""
    doc = nlp(text)
    sentences = list(doc.sents)
    
    paragraphs = []
    current_para = []
    current_word_count = 0
    
    for sent in sentences:
        sent_text = sent.text.strip()
        sent_words = len([token for token in sent if token.is_alpha])
        
        if current_word_count + sent_words > max_words and current_para:
            # Save current paragraph if it meets minimum
            if current_word_count >= min_words:
                para_text = ' '.join(current_para)
                paragraphs.append(para_text)
            current_para = [sent_text]
            current_word_count = sent_words
        else:
            current_para.append(sent_text)
            current_word_count += sent_words
    
    # Add last paragraph
    if current_para and current_word_count >= min_words:
        para_text = ' '.join(current_para)
        paragraphs.append(para_text)
    
    return paragraphs


def load_data_from_folder(folder_path, max_paragraphs=None):
    """Load and split text files into paragraphs"""
    all_paragraphs = []
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    print(f"Processing {len(files)} files from {folder_path}")
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        paragraphs = split_into_paragraphs(text, min_words=150, max_words=300)
        all_paragraphs.extend(paragraphs)
        print(f"  {filename}: {len(paragraphs)} paragraphs")
    
    if max_paragraphs and len(all_paragraphs) > max_paragraphs:
        # Randomly sample
        indices = np.random.choice(len(all_paragraphs), max_paragraphs, replace=False)
        all_paragraphs = [all_paragraphs[i] for i in indices]
    
    print(f"Total paragraphs from {folder_path}: {len(all_paragraphs)}")
    return all_paragraphs


class TextClassificationDataset(Dataset):
    """Dataset for text classification with DistilBERT"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate_epoch(model, data_loader, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (AI)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1, np.array(true_labels), np.array(predictions), np.array(all_probs)


def main():
    print("="*80)
    print("DistilBERT + LoRA Fine-tuning for Human vs AI Text Classification")
    print("="*80)
    
    # Define paths
    base_path = '/home/mohit/projects/precog_task/datasets'
    class_1_path = os.path.join(base_path, 'class_1')
    class_2_path = os.path.join(base_path, 'class_2')
    class_3_path = os.path.join(base_path, 'class_3')
    output_dir = '/home/mohit/projects/precog_task/task_2/tier_c'
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    print("-" * 80)
    
    # Load human paragraphs from class_1
    print("\nLoading Class 1 (Human text)...")
    class_1_all = load_data_from_folder(class_1_path, max_paragraphs=None)
    
    # Load AI paragraphs from class_2 and class_3
    print("\nLoading Class 2 (AI text)...")
    class_2_paragraphs = load_data_from_folder(class_2_path, max_paragraphs=None)
    
    print("\nLoading Class 3 (AI Mimic text)...")
    class_3_paragraphs = load_data_from_folder(class_3_path, max_paragraphs=None)
    
    # Combine AI paragraphs (both class_2 and class_3 are labeled as AI = 1)
    ai_paragraphs = class_2_paragraphs + class_3_paragraphs
    print(f"\nTotal AI paragraphs (class_2 + class_3): {len(ai_paragraphs)}")
    
    # Step 2: Split data
    print("\n[Step 2] Splitting data into train/val/test...")
    print("-" * 80)
    
    # Shuffle data
    np.random.shuffle(class_1_all)
    np.random.shuffle(ai_paragraphs)
    
    # Use ~600 human paragraphs for training, 20 val, 50 test
    human_train_size = 600
    human_val_size = 20
    human_test_size = 50
    
    human_train = class_1_all[:human_train_size]
    human_val = class_1_all[human_train_size:human_train_size + human_val_size]
    human_test = class_1_all[human_train_size + human_val_size:human_train_size + human_val_size + human_test_size]
    
    # Use ALL AI paragraphs, split proportionally
    total_ai = len(ai_paragraphs)
    ai_train_size = min(600, int(total_ai * 0.85))  # ~85% for training
    ai_val_size = min(20, int(total_ai * 0.075))    # ~7.5% for validation
    ai_test_size = total_ai - ai_train_size - ai_val_size  # Rest for testing
    
    ai_train = ai_paragraphs[:ai_train_size]
    ai_val = ai_paragraphs[ai_train_size:ai_train_size + ai_val_size]
    ai_test = ai_paragraphs[ai_train_size + ai_val_size:]
    
    # Combine and create labels
    train_texts = human_train + ai_train
    train_labels = [0] * len(human_train) + [1] * len(ai_train)
    
    val_texts = human_val + ai_val
    val_labels = [0] * len(human_val) + [1] * len(ai_val)
    
    test_texts = human_test + ai_test
    test_labels = [0] * len(human_test) + [1] * len(ai_test)
    
    # Shuffle training data
    train_indices = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in train_indices]
    train_labels = [train_labels[i] for i in train_indices]
    
    print(f"Training set: {len(train_texts)} samples ({train_labels.count(0)} human, {train_labels.count(1)} AI)")
    print(f"Validation set: {len(val_texts)} samples ({val_labels.count(0)} human, {val_labels.count(1)} AI)")
    print(f"Test set: {len(test_texts)} samples ({test_labels.count(0)} human, {test_labels.count(1)} AI)")
    
    # Save data splits
    data_splits = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'val_texts': val_texts,
        'val_labels': val_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }
    splits_path = os.path.join(output_dir, 'data_splits.pkl')
    with open(splits_path, 'wb') as f:
        pickle.dump(data_splits, f)
    print(f"✓ Data splits saved to: {splits_path}")
    
    # Step 3: Initialize tokenizer
    print("\n[Step 3] Initializing DistilBERT tokenizer...")
    print("-" * 80)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Step 4: Create datasets
    print("\n[Step 4] Creating datasets...")
    print("-" * 80)
    
    max_length = 256
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    print(f"✓ Datasets created (max_length={max_length})")
    
    # Step 5: Create dataloaders
    print("\n[Step 5] Creating dataloaders...")
    print("-" * 80)
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ DataLoaders created (batch_size={batch_size})")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 6: Initialize model with LoRA
    print("\n[Step 6] Initializing DistilBERT model with LoRA...")
    print("-" * 80)
    
    # Load base model
    base_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # Apply LoRA to query and value projections
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized with LoRA")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    print(f"\nLoRA configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    # Step 7: Setup training
    print("\n[Step 7] Setting up training...")
    print("-" * 80)
    
    epochs = 3
    learning_rate = 3e-4
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ Training setup complete")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Step 8: Train model
    print("\n[Step 8] Training model...")
    print("-" * 80)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model (Val F1: {val_f1:.4f})")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model (Val F1: {best_val_f1:.4f})")
    
    # Save model
    model_path = os.path.join(output_dir, 'distilbert_lora_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Model and tokenizer saved to: {model_path}")
    
    # Step 9: Evaluate on test set
    print("\n[Step 9] Evaluating on test set...")
    print("-" * 80)
    
    test_loss, test_acc, test_f1, test_labels_np, test_predictions, test_probs = evaluate_epoch(model, test_loader, device)
    
    # Calculate additional metrics
    test_precision = precision_score(test_labels_np, test_predictions)
    test_recall = recall_score(test_labels_np, test_predictions)
    test_auc = roc_auc_score(test_labels_np, test_probs)
    
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"\nAccuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(test_labels_np, test_predictions, 
                                target_names=['Human', 'AI'], digits=4))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(test_labels_np, test_predictions)
    print(cm)
    print("\nConfusion Matrix Interpretation:")
    print("             Predicted:")
    print("             Human    AI")
    print(f"Actual Human:   {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"Actual AI:      {cm[1][0]:3d}    {cm[1][1]:3d}")
    
    # Step 10: Save plots
    print("\n[Step 10] Saving training plots...")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 plot
    axes[1, 0].plot(val_f1s, label='Val F1-Score', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_labels_np, test_probs)
    axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_auc:.4f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[1, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1, 1].set_title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {plot_path}")
    
    # Step 11: Save results
    print("\n[Step 11] Saving results...")
    print("-" * 80)
    
    # Save predictions
    results_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels_np,
        'predicted_label': test_predictions,
        'probability_ai': test_probs,
        'true_class': ['Human' if l == 0 else 'AI' for l in test_labels_np],
        'predicted_class': ['Human' if l == 0 else 'AI' for l in test_predictions]
    })
    results_path = os.path.join(output_dir, 'test_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Test predictions saved to: {results_path}")
    
    # Save metrics
    results = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_accs': [float(x) for x in train_accs],
        'val_accs': [float(x) for x in val_accs],
        'val_f1s': [float(x) for x in val_f1s],
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(test_labels_np, test_predictions, 
                                                       target_names=['Human', 'AI'], 
                                                       output_dict=True)
    }
    
    results_pkl_path = os.path.join(output_dir, 'results.pkl')
    with open(results_pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to: {results_pkl_path}")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print("\nSummary of saved files:")
    print(f"  - Model: {model_path}")
    print(f"  - Data splits: {splits_path}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - Test predictions: {results_path}")
    print(f"  - Results: {results_pkl_path}")
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC-ROC: {test_auc:.4f}")


if __name__ == '__main__':
    main()
