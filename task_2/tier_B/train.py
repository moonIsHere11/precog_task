import os
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pickle

print("Loading spacy model...")
nlp = spacy.load('en_core_web_sm')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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


def text_to_embedding(text, fasttext_model, embedding_dim=300):
    """Convert text to FastText embedding by averaging word vectors (including OOV via subwords)"""
    doc = nlp(text.lower())
    vectors = []
    
    for token in doc:
        if token.is_alpha:
            # FastText handles OOV words via subword information
            try:
                vectors.append(fasttext_model[token.text])
            except KeyError:
                # If token not in vocab, FastText will use subword info
                pass
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    
    return np.mean(vectors, axis=0)


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
            
            nn.Linear(64, 1)  # No sigmoid - using BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=10):
    """Train the model with early stopping"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Apply sigmoid to get probabilities, then threshold
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                
                outputs = model(embeddings).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                # Apply sigmoid to get probabilities, then threshold
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            outputs = model(embeddings).squeeze()
            probs = torch.sigmoid(outputs)  # Apply sigmoid for probabilities
            predictions = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    return all_labels, all_predictions, all_probs


def main():
    print("="*80)
    print("Feedforward Neural Network for Human vs AI Text Classification")
    print("="*80)
    
    # Define paths
    base_path = '/home/mohit/projects/precog_task/datasets'
    class_1_path = os.path.join(base_path, 'class_1')
    class_2_path = os.path.join(base_path, 'class_2')
    class_3_path = os.path.join(base_path, 'class_3')
    output_dir = '/home/mohit/projects/precog_task/task_2/tier_B'
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    print("-" * 80)
    
    # Load ~600 human paragraphs from class_1
    print("\nLoading Class 1 (Human text)...")
    class_1_all = load_data_from_folder(class_1_path, max_paragraphs=None)
    
    # Randomly shuffle and select ~600 paragraphs for training + val + test
    np.random.shuffle(class_1_all)
    total_human = min(len(class_1_all), 700)  # Use up to 700 if available
    class_1_paragraphs = class_1_all[:total_human]
    print(f"Selected {len(class_1_paragraphs)} human paragraphs")
    
    # Load ALL AI paragraphs from class_2 and class_3
    print("\nLoading Class 2 (AI text)...")
    class_2_paragraphs = load_data_from_folder(class_2_path, max_paragraphs=None)
    
    print("\nLoading Class 3 (AI Mimic text)...")
    class_3_paragraphs = load_data_from_folder(class_3_path, max_paragraphs=None)
    
    # Combine ALL AI paragraphs (both class_2 and class_3 are labeled as AI = 1)
    ai_paragraphs = class_2_paragraphs + class_3_paragraphs
    print(f"\nTotal AI paragraphs (class_2 + class_3): {len(ai_paragraphs)}")
    
    # Step 2: Split data
    print("\n[Step 2] Splitting data into train/val/test...")
    print("-" * 80)
    
    # Split human paragraphs: ~85% train, ~7.5% val, ~7.5% test
    total_human = len(class_1_paragraphs)
    human_test_size = max(20, int(0.1 * total_human))  # At least 20 for test
    human_val_size = max(20, int(0.1 * total_human))   # At least 20 for val
    human_train_size = total_human - human_test_size - human_val_size
    
    human_train = class_1_paragraphs[:human_train_size]
    human_val = class_1_paragraphs[human_train_size:human_train_size + human_val_size]
    human_test = class_1_paragraphs[human_train_size + human_val_size:]
    
    # Split ALL AI paragraphs proportionally
    total_ai = len(ai_paragraphs)
    ai_test_size = max(20, int(0.1 * total_ai))  # ~10% for test, at least 20
    ai_val_size = max(20, int(0.1 * total_ai))   # ~10% for val, at least 20
    ai_train_size = total_ai - ai_test_size - ai_val_size
    
    np.random.shuffle(ai_paragraphs)
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
    
    # Step 3: Load FastText model (already downloaded)
    print("\n[Step 3] Loading FastText pre-trained model...")
    print("-" * 80)
    print("Loading pre-downloaded model from gensim cache...")
    
    # Load from gensim's data directory
    import gensim.downloader as api
    fasttext_model = api.load('fasttext-wiki-news-subwords-300')
    embedding_dim = 300
    print(f"✓ FastText model loaded (dimension: {embedding_dim})")
    
    # Step 4: Generate embeddings
    print("\n[Step 4] Generating embeddings...")
    print("-" * 80)
    
    print("Generating training embeddings...")
    train_embeddings = []
    for i, text in enumerate(train_texts):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(train_texts)} texts...")
        emb = text_to_embedding(text, fasttext_model, embedding_dim)
        train_embeddings.append(emb)
    train_embeddings = np.array(train_embeddings)
    
    print("Generating validation embeddings...")
    val_embeddings = []
    for text in val_texts:
        emb = text_to_embedding(text, fasttext_model, embedding_dim)
        val_embeddings.append(emb)
    val_embeddings = np.array(val_embeddings)
    
    print("Generating test embeddings...")
    test_embeddings = []
    for text in test_texts:
        emb = text_to_embedding(text, fasttext_model, embedding_dim)
        test_embeddings.append(emb)
    test_embeddings = np.array(test_embeddings)
    
    print(f"✓ Embeddings generated:")
    print(f"  Train: {train_embeddings.shape}")
    print(f"  Val: {val_embeddings.shape}")
    print(f"  Test: {test_embeddings.shape}")
    
    # Step 4.5: Normalize embeddings
    print("\n[Step 4.5] Normalizing embeddings...")
    print("-" * 80)
    train_embeddings = normalize(train_embeddings, norm='l2', axis=1)
    val_embeddings = normalize(val_embeddings, norm='l2', axis=1)
    test_embeddings = normalize(test_embeddings, norm='l2', axis=1)
    print("✓ Embeddings normalized (L2 norm)")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'embeddings.npz')
    np.savez(embeddings_path,
             train_embeddings=train_embeddings,
             train_labels=np.array(train_labels),
             val_embeddings=val_embeddings,
             val_labels=np.array(val_labels),
             test_embeddings=test_embeddings,
             test_labels=np.array(test_labels))
    print(f"✓ Embeddings saved to: {embeddings_path}")
    
    # Save texts for reference
    texts_data = {
        'train_texts': train_texts,
        'val_texts': val_texts,
        'test_texts': test_texts
    }
    texts_path = os.path.join(output_dir, 'texts.pkl')
    with open(texts_path, 'wb') as f:
        pickle.dump(texts_data, f)
    print(f"✓ Texts saved to: {texts_path}")
    
    # Step 5: Create DataLoaders
    print("\n[Step 5] Creating DataLoaders...")
    print("-" * 80)
    
    train_dataset = TextDataset(train_embeddings, train_labels)
    val_dataset = TextDataset(val_embeddings, val_labels)
    test_dataset = TextDataset(test_embeddings, test_labels)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ DataLoaders created (batch_size={batch_size})")
    
    # Step 6: Initialize model
    print("\n[Step 6] Initializing model...")
    print("-" * 80)
    
    model = FeedForwardNN(input_dim=embedding_dim).to(device)
    
    # Calculate positive weight for class imbalance
    num_negatives = train_labels.count(0)  # Human
    num_positives = train_labels.count(1)  # AI
    pos_weight = torch.tensor([num_negatives / num_positives]).to(device)
    print(f"Class balance - Human: {num_negatives}, AI: {num_positives}")
    print(f"Positive weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Step 7: Train model
    print("\n[Step 7] Training model...")
    print("-" * 80)
    
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        epochs=50, patience=10
    )
    
    # Save model
    model_path = os.path.join(output_dir, 'nn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Step 8: Evaluate on test set
    print("\n[Step 8] Evaluating on test set...")
    print("-" * 80)
    
    test_labels_np, test_predictions, test_probs = evaluate_model(model, test_loader)
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels_np, test_predictions)
    test_auc = roc_auc_score(test_labels_np, test_probs)
    test_f1 = f1_score(test_labels_np, test_predictions)
    test_precision = precision_score(test_labels_np, test_predictions)
    test_recall = recall_score(test_labels_np, test_predictions)
    
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"\nAccuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"AUC-ROC:   {test_auc:.4f}")
    
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
    
    # Step 9: Plot training history
    print("\n[Step 9] Saving training plots...")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {plot_path}")
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels_np, test_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {roc_path}")
    
    # Save results summary
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
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(test_labels_np, test_predictions, 
                                                       target_names=['Human', 'AI'], 
                                                       output_dict=True)
    }
    
    results_path = os.path.join(output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print("\nSummary of saved files:")
    print(f"  - Model weights: {model_path}")
    print(f"  - Embeddings: {embeddings_path}")
    print(f"  - Texts: {texts_path}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - ROC curve: {roc_path}")
    print(f"  - Results: {results_path}")


if __name__ == '__main__':
    main()
