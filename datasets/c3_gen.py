from google import genai
from google.genai import types
import os

TOPIC_KEYWORDS = "desire,actions,instinct,animals"
AUTHOR = "Bertrand Russell"
WORK = "The Analysis of Mind"
output_file = "./b1m_t10.txt"

suffix_prompt = (
    f"Write a 2000 word essay of clean text with no subheadings or headings.(only return the essay and nothing else) .The essay should closely mimic the style of the author {AUTHOR}, the style is as follows : Tone-Conversational with professional undertones Writing Style- Formal, Persuasive   Voice- Authoritative and objective  Language Level- Highly Formal,  complex."
)

references = ( 
    f"Here are a few references to the writings of {AUTHOR} (on an unrelated topic), examine the style and try to replicate it."
    f"Example1: Guild Socialists, as we have seen, have another suggestion, growing naturally out of the autonomy of industrial guilds, by which they hope to limit the power of the State and help to preserve individual liberty. They propose that, in addition to Parliament, elected (as at present) on a territorial basis and representing the community as consumers, there shall also be a ``Guild Congress,'' a glorified successor of the present Trade Union Congress, which shall consist of representatives chosen by the Guilds, and shall represent the community as producers. This method of diminishing the excessive power of the State has been attractively set forth by Mr. G. D. H. Cole in his ``Self-Government in Industry.'' ``Where now,'' he says, ``the State passes a Factory Act, or a Coal Mines Regulation Act, the Guild Congress of the future will pass such Acts, and its power of enforcing them will be the same as that of the State'.His ultimate ground for advocating this system is that, in his opinion, it will tend to preserve individual liberty: ``The fundamental reason for the preservation, in a democratic Society, of both the industrial and the political forms of Social organization is, it seems to me, that only by dividing the vast power now wielded by industrial capitalism can the individual hope to be free"
    f"Example2:At present, the capitalist has more control over the lives of others than any man ought to have; his friends have authority in the State; his economic power is the pattern for political power. In a world where all men and women enjoy economic freedom, there will not be the same habit of command, nor, consequently, the same love of despotism; a gentler type of character than that now prevalent will gradually grow up. Men are formed by their circumstances, not born ready- made. The bad effect of the present economic system on character, and the immensely better effect to be expected from communal ownership, are among the strongest reasons for advocating the change. In the world as we have been imagining fit, economic fear and most economic hope will be alike removed out of life. No one will be haunted by the dread of poverty or driven into ruthlessness by the hope of wealth. There will not be the distinction of social classes which now plays such an immense part in life. The unsuccessful professional man will not live in terror lest his children should sink in the scale; the aspiring employe will not be looking forward to the day when he can become a sweater in his turn. Ambitious young men will have to dream other daydreams than that of business success and wealth wrung out of the ruin of competitors and the degradation of labor. In such a world, most of the nightmares that lurk in the background of men's minds will no longer exist; on the other hand, ambition and the desire to excel will have to take nobler forms than those that are encouraged by a commercial society. All those activities that really confer benefits upon mankind will be open, not only to the fortunate few, but to all who have sufficient ambition and native aptitude. Science, labor-saving inventions, technical progress of all kinds, may be confidently expected to flourish far more than at present, since they will be the road to honor, and honor will have to replace money among those of the young who desire to achieve success."
)

#define each of the angles 
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

def generate_and_save(prompt_text):   
    config = types.GenerateContentConfig(
            max_output_tokens=8192, 
            temperature=0.7,   # basically giving it more flexibility and creativeness  
    )
    
    client = genai.Client(api_key="AIzaSyD8gDwI4qO9A0-f8_MdwVV8")  #not the API key that i used, just a placeholder 
    
    model_id = "gemini-2.5-flash-lite"  #we can keep switching model once RPD limits are hit
    
    print(f"Sending prompt to {model_id}")

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_text
        )
        

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(response.text)
            f.write("\n\nEND\n\n") # Append the separator
        
        print(f"Success! Response added to {output_file}")
        

    except Exception as e:
        print(f"An error occurred while generating {output_file}")

if __name__ == "__main__":
    print()
    for i, angle_prompt in enumerate(angles, start=1):         # Combine the angle with the formatting suffix
        full_prompt = angle_prompt + suffix_prompt + references
        
        print()
        generate_and_save(full_prompt)