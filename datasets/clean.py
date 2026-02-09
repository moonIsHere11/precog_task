import nltk
import re
import os


try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def clean_noise(text):
    # 1. Remove everything in square brackets [Footnotes, citations, references]
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    
    # Remove standalone lines in ALL CAPS (Chapter titles, headers, subheadings)
    # This targets lines that consist only of uppercase, numbers, and punctuation.
    text = re.sub(r'(?m)^\s*[A-Z0-9\s.,;:"\'!?-]+\s*$', '', text)
    
    #Targeted CHAPTER/Section removal (Case-insensitive catch-all)
    text = re.sub(r'(?m)^\s*(CHAPTER|Section|Book|Part|Note)\s+[IVXLCDM\d]+\s*$', '', text, flags=re.I)
    
    # Collapse random mid-sentence newlines (Normalizes the text stream)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_author_text(input_file, output_file, min_words=100, max_words=200):
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found. Skipping.")
        return 0

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    text = clean_noise(text)

    sentences = nltk.sent_tokenize(text)
    
    final_paragraphs = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        count = len(words)
        
        # Discard small sentences 
        if count < 8:
            continue

        if current_word_count + count > max_words:
            # ONLY save if it satisfies the minimum word requirement
            if current_word_count >= min_words:
                final_paragraphs.append(" ".join(current_chunk))
            
            if count > max_words:
                current_chunk = words[:max_words]
                current_word_count = max_words
            else:
                current_chunk = words
                current_word_count = count
        else:
            current_chunk.extend(words)
            current_word_count += count
            
    # Final cleanup of the trailing chunk
    if min_words <= current_word_count <= max_words:
        final_paragraphs.append(" ".join(current_chunk))

    with open(output_file, 'w', encoding='utf-8') as f:
        for para in final_paragraphs:
            f.write(para + "\n\n")
            
    return len(final_paragraphs)

# Execution Block
files_to_process = {
    "author_a.txt": "mill_subjection_new.txt",
    "author_A2.txt": "mill_liberty_new.txt",
    "author_b.txt": "russell_roads_new.txt",
    "author_B2.txt": "russell_mind_new.txt"
}

print("Starting Professional Data Scouring & Segmentation...")
results = {}

for inp, out in files_to_process.items():
    count = segment_author_text(inp, out)
    results[inp] = count
    print(f"Processed {inp} -> {count} valid paragraphs.")

mill_total = results.get("author_a.txt", 0) + results.get("author_A2.txt", 0)
russell_total = results.get("author_b.txt", 0) + results.get("author_B2.txt", 0)

print("\nCLASS 1 FINAL AUDIT:")
print(f"John Stuart Mill: {mill_total} paragraphs")
print(f"Bertrand Russell: {russell_total} paragraphs")
print(f"Grand Total Human Samples: {mill_total + russell_total}")