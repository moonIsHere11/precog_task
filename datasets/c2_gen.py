from google import genai
import os

TOPIC_KEYWORDS = "desire,actions,instinct,animals"
AUTHOR = "Bertrand Russel"
WORK = "Analysis of the Mind"


suffix_prompt = (
    "\n\nThe essay is NOT about analysing the book, rather its about the topic. "
    "So although you can draw inspiration for the flow of topics from the book "
    "DO NOT refer to it. Write a 2000 word essay of clean text with no subheadings or headings."
)

#different angles of explanation 
angles = [
    # 1. Analytical
    f"Analyze the topic of {TOPIC_KEYWORDS} in the context of {AUTHOR}’s {WORK}. "
    f"Conduct the analysis in a systematic and explanatory manner. Break the subject into its core components, "
    f"underlying mechanisms, and key relationships. Explain how and why these elements interact, including "
    f"relevant historical context, conceptual frameworks, and causal chains. Prioritize clarity, logical structure, "
    f"and precision over persuasion or storytelling. Avoid moral judgments or personal opinions. "
    f"Assume an intelligent reader seeking deep understanding rather than advocacy.",

    # 2. Clarification based
    f"Clarify the conceptual meaning of {TOPIC_KEYWORDS} in the context of {AUTHOR}’s {WORK}. "
    f"Identify key terms, implicit assumptions, and conceptual distinctions necessary for understanding the topic. "
    f"Explain how these concepts are defined, constrained, or qualified within the work, and how they relate to "
    f"one another. Emphasize conceptual precision and interpretive clarity rather than evaluation or argument.",

    # 3. Causal
    f"Explain the causal structure underlying {TOPIC_KEYWORDS} in the context of {AUTHOR}’s {WORK}. "
    f"Identify the social, psychological, legal, or institutional causes discussed or implied in the text, "
    f"and trace their effects systematically. Describe the direction of influence, intermediate mechanisms, "
    f"and long-term consequences. Focus on explanation rather than judgment or recommendation.",

    # 4. Structural
    f"Examine {TOPIC_KEYWORDS} in the context of {AUTHOR}’s {WORK} from a system-level perspective. "
    f"Describe the institutional, legal, and social structures involved, and explain how these structures "
    f"constrain or shape individual behavior and character. Highlight interactions between different levels "
    f"of organization (individual, social, institutional) without advocating for reform or expressing evaluative positions."
]

def generate_and_save(prompt_text, filename):
    client = genai.Client(api_key="AIzaSyD8gDwI4qO9A0oigT2-f8_MdwVV8")  # this one here is a modified API key, not mine. (For safety reasons T_T)
    
    model_id = "gemini-3-flash-preview"
    
    print(f"Sending prompt to {model_id} for {filename}...")
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_text
        )
        
        # Open in "a" (append) mode
        output_file = "./b1_t10.txt"
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(response.text)
            f.write("\n\nEND\n\n") # Append the separator
        
        print(f"Success! Response added to {output_file}")
        

    except Exception as e:
        print(f"An error occurred while generating {filename}: {e}")

if __name__ == "__main__":
    print()
    for i, angle_prompt in enumerate(angles, start=1):         # Combine the angle prompt with the formatting suffix
        full_prompt = angle_prompt + suffix_prompt
        
        target_filename = f"gemini_response_a1_angle_{i}.md"  # i cleaned and moved them to 
        
        # Execute the generation
        generate_and_save(full_prompt, target_filename)