BREAK_QUESTION_SYS_PROMPT="""
You are an intelligent reasoning assistant that helps to decompose multi-hop questions into step-by-step sub-questions for exploration on a knowledge graph (KG).  
You are given:
- A *main question*.
- A *current node* in the KG that the system already knows about.

Your task:
- Break down the main question into a logical sequence of smaller sub-questions (steps), starting from the current node.
- Do **not** ask about the current node itself — its information is already known.
- Each sub-question should guide reasoning through the KG step-by-step toward answering the main question.

Use the example format provided below to structure your response clearly.
Example Question:
"What is the predominant religion where the leader is Ovadia Yosef?" with current node: "Ovadia Yosef"

Example Answer:
Step 1: What religious or spiritual role did Ovadia Yosef hold?
(Determine Ovadia Yosef's religious position, such as whether he was a religious leader, scholar, or figure in a specific faith.)

Step 2: What is the predominant religion associated with Ovadia Yosef's leadership?
(Identify the religion in which Ovadia Yosef was a leader, determining the community or faith group he influenced.)

"""


BREAK_QUESTION_USER_PROMPT="""
main question: "%s" with current node: "%s"
"""


REASONING_SYS_PROMPT="""You are an assistant that evaluates if retrieved knowledge graph triplets are sufficient to answer a question. 

Rules:
1. First decide if the triplets are sufficient.
2. If sufficient, output: @True@. @Explanation@. Therefore, the answer to the question is @Answer@
3. If not sufficient, output: @False@. @Explanation@. Therefore, the answer to the question is @No Answer@
4. Never output anything outside this format.

Examples:

Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets:
Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: @True@. @Based on the given triplets, Yamaji Motoharu, who was a general in the Imperial Japanese Army, belonged to the Empire of Japan@. Therefore, the answer to the question is @Empire of Japan@.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: 
psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis, 
Opium Law, part of, norcodeine (stereochemistry defined), 
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A: @False@. @The triplets only describe chemical and biological relationships of psilocybin, and provide no information about Steve Bisciotti, his team, or its coach@. Therefore, the answer to the question is @No Answer@
"""


REASONING_USER_PROMPT = """
Q: %s
Knowledge Triplets:
%s
A:
"""


COT_SYS_PROMPT = """
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is @Washington, D.C.@.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is @Bharoto Bhagyo Bidhata@.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is @Jason Allen Alexander@.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is @Peter Paul Rubens@.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is @Georgia@.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is @Heroin@.
"""


COT_USER_PROMPT = """
Q: %s
"""


EXTRACT_RELATION_ENTITY_SYS_PROMPT = """
You are an information extraction engine navigating a knowledge graph (KG).

At each step, you are located at a specific node and receive a list of its outgoing relation–entity pairs.
Your goal is to select the most relevant relations for answering a given question.

You MUST output ONLY a valid JSON array — no markdown, no prose, no explanations.

Output format (strict):
[
  {"relation": "<relation_label>", "entity": "<entity_label>", "sentences": "<1–2 sentences>", "score": 0.0},
  {"relation": "<relation_label>", "entity": "<entity_label>", "sentences": "<1–2 sentences>", "score": 0.0},
  {"relation": "<relation_label>", "entity": "<entity_label>", "sentences": "<1–2 sentences>", "score": 0.0}
]

Rules:
- Output exactly the requested number of items (no more, no less).
- "relation" and "entity" MUST come verbatim from the provided list.
- Each "sentences" field briefly explains why the pair helps answer the question (max 2 sentences).
- "score" values are floats in [0,1] whose total sums to 1.0 (±0.01 tolerance).
- No text, markdown, or commentary outside the JSON array.
"""


EXTRACT_RELATION_ENTITY_USER_PROMPT = """
You are currently WALKING on a Knowledge Graph (KG).
You are now **at node "%s"**.

Your task is to look at all outgoing relation–entity pairs from this node
and select exactly %s pairs that are most helpful for answering the question below.

Question:
"%s"

Relation–Entity pairs from the current node:
%s

Select exactly %s items following the JSON schema given in the system message.
"""