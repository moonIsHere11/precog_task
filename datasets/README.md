   ## Authors / source files

   John Stuart Mill
       The Subjection of Women (A1)
       On Liberty (A2)

   Bertrand Russell
       Proposed Roads to Freedom (B1)
       Analysis of the Mind (B2)

   Picked Authors from different eras and view points to have some diversity of style, so that the model doesnt overfit onto a specific way of writing. The books were picked such that they have a good amount of overlap and topic cannot be a very obvious giveaway.    

   ## Dataset cleaning

   Below are the steps used to clean the raw texts before producing the final dataset:

   1. Removed Project Gutenberg headers and tails

   2. Maintained paragraph lengths in the 100–200 word range
      - Paragraphs longer than 200 words were split into contiguous chunks of approximately 100–200 words so each sample falls within the target range.

   3. Removed surface features and non-content markers
      - References to explicit chapter/section markers such as lines containing "SECTION X" or "CHAPTER X" were removed. (Regex)
      - Footnotes were removed. (Regex)
      - Text and references enclosed in square brackets (e.g. "[see note]" or citation markers like "[1]") were removed. (Regex)
      - Asterisk ("*") before the reference explanations were removed.(Find and Replace Tool)


## Class 2 dataset generation 
   - Paragraphs from the novels are usually a mixture of different topics/views that the author talks about. 
   - To extract these specific and fine topics, BERTopic was used. (uses c-TF-IDF to identify and output a list of highly specific and corelated words that are the closest to depicting the meaning of the input text) 
   (Refer to topic_ext jupyter notebook for implementation)


   - #### Topics found 
      Book A1                   
      a. human_power_men_character
      b. women_men_things_great
      d. women_men_society_power

      Book A2
      e. society_liberty_human_conduct
      f. truth_opinion_discussions

      Book A3
      g. work_present_socialism_state
      h. class_marx_syndicalism_labor

      Book A4
      i. images_past_memory_sensations
      j. object_matter_mental_physical
      k. desire_actions_instinct_animals

   2. #### Limitations: 
       - Gemini API w/o batching only gives us 20 Requests per day
       - Prompting for 500 paragraphs on the same specific topics results in highly repetitive results. They also lack long term context and feel very much like a short paragraph and not like a part of a novel.

       Approach: 
       - Have gemini generate long essays on specific topics giving a more novel like feel and split them into 100-200 word chunks. 
       - To solve repetitiveness, prompt gemini to approach the topic from different angles: Analytical, Clarification heavy, Casual and Structural. 
       (Specific prompts can be seen in c2_gen.py)
       
      To check if the LLM responses sticked to the topic, embeddings were generated for the inital few responses and cosine similarity was calculated. 
      For few picked examples, got values around 0.65 which is a reasonable level of similarity.  


## Class 3 dataset generation 
- How do you let the LLM copy the authors style without directly showing the essay/ giving rigid mathematical features like TTR etc for it to abide by? 
- On exploring through r/writing on ways people try to mimic an author, i came across this tool. 
  https://www.superwriter.io/writing-style-checker 
  Descibes style across fields like Language level, Writing Style, Tone, Voice of narration etc. 
  (Pretty popular post with quite a lot of people saying it was good which is why i went forward with this)

- For every prompt, 
   1. extract the features from the above website. 
   2. include random paragraphs from the actual text, for a better understanding of style. 
   3. Mention Author, Book Name etc (same as Class 2)



## Directory Structure

```text
datasets/
├── README.md
├── ai_human_ds.csv        # testing data was sampled from here
├── c2_gen.py              # Script for Class 2 generation
├── c3_gen.py              # Script for Class 3 generation
├── clean.py               # Script for Class 1 cleaning
├── gtest_data.csv         # Small test dataset
├── topic_ext.ipynb        # Topic extraction notebook
├── class_1/               # Cleaned Human Data
│   ├── a1.txt
│   ├── a2.txt
│   ├── b1.txt
│   └── b2.txt
├── class_2/               # Basic AI Generated Data (across 10 topics)
│   ├── a1_t1.txt
│   ├── a1_t2.txt
│   ├── a1_t3.txt
│   ├── a2_t4.txt
│   ├── a2_t5.txt
│   ├── b1_t6.txt
│   ├── b1_t7.txt
│   ├── b1_t8.txt
│   ├── b1_t9.txt
│   ├── b1_t10.txt
│   └── gen_test.ipynb
├── class_3/               # Mimic AI Generated Data
│   ├── a1m_t1.txt
│   ├── a1m_t2.txt
│   ├── a1m_t3.txt
│   ├── a2m_t4.txt
│   ├── a2m_t5.txt
│   ├── b1m_t6.txt
│   ├── b1m_t7.txt
│   ├── b2m_t8.txt
│   ├── b2m_t9.txt
│   └── b2m_t10.txt
└── raw_data/              # Original Project Gutenberg Texts
    ├── author_a.txt
    ├── author_A2.txt
    ├── author_b.txt
    └── author_B2.txt
```
