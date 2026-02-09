import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load base model and LoRA weights
print("Loading fine-tuned model...")
base_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
model = PeftModel.from_pretrained(
    base_model, 
    '/home/mohit/projects/precog_task/task_2/tier_c/distilbert_lora_model'
)
model = model.to(device)
model.eval()
print("Model loaded!\n")

def predict_text(text):
    """Make prediction on a single text"""
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1)
    
    # Get results
    pred_label = prediction.item()
    prob_human = probs[0][0].item()
    prob_ai = probs[0][1].item()
    
    return pred_label, prob_human, prob_ai

print("="*80)
print("DistilBERT Human vs AI Text Classifier - Interactive Mode")
print("="*80)
print("Commands:")
print("  - Type or paste your text and press Enter")
print("  - Type 'quit' or 'exit' to stop")
print("  - Type 'test' to run some sample texts")
print("="*80)

# Test samples
test_samples = {
    "human_1": "I walked to the store yesterday and bought some groceries. It was a nice day outside, so I took my time.",
    "human_2": "Honestly, I think that movie was overrated. The plot didn't make sense and the acting was mediocre at best.",
    "ai_1": "The implementation of artificial intelligence in modern healthcare systems has revolutionized diagnostic procedures. Machine learning algorithms can now analyze medical imaging with unprecedented accuracy.",
    "ai_2": "In conclusion, the data demonstrates a clear correlation between the variables studied. Further research is warranted to explore the underlying mechanisms.",
}

while True:
    print("\n" + "-"*80)
    user_input = input("\nEnter text (or 'quit'/'test'): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nExiting...")
        break
    
    if user_input.lower() == 'test':
        print("\nRunning test samples:\n")
        for name, text in test_samples.items():
            pred_label, prob_human, prob_ai = predict_text(text)
            label_name = "HUMAN" if pred_label == 0 else "AI"
            
            print(f"\n[{name}]")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {label_name}")
            print(f"Confidence: Human={prob_human:.4f}, AI={prob_ai:.4f}")
        continue
    
    if not user_input:
        print("Please enter some text.")
        continue
    
    # Make prediction
    pred_label, prob_human, prob_ai = predict_text(user_input)
    label_name = "HUMAN" if pred_label == 0 else "AI"
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"Text length: {len(user_input)} characters")
    print(f"\nPrediction: {label_name}")
    print(f"Confidence:")
    print(f"  Human: {prob_human:.4f} ({prob_human*100:.2f}%)")
    print(f"  AI:    {prob_ai:.4f} ({prob_ai*100:.2f}%)")
    print("="*80)
