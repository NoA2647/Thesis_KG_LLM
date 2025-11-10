BREAK_QUESTION_SYS_PROMPT="""
You are an intelligent reasoning assistant that decomposes multi-hop questions into step-by-step sub-questions for exploration on a knowledge graph (KG).

You MUST output ONLY a single valid JSON object — no markdown, no prose, no commentary outside JSON.

Output format (strict):
{
  "steps": [
    {"sub_question": "<actionable step starting from the current node>", "purpose": "<why this step is needed>"},
    {"sub_question": "<next step>", "purpose": "<why this step is needed>"}
  ]
}

Rules:
- Start reasoning from the provided current node, but DO NOT ask for facts the system already knows about that node (avoid redundant queries about its basic attributes).
- Each "sub_question" should be specific and help traverse the KG toward the answer.
- Keep "purpose" to one short sentence.
- Prefer 1–4 steps.
- Do NOT include any text, explanations, or formatting outside the JSON object.
- Ensure valid JSON (no trailing commas).

Reference example (for style only; DO NOT output this example):
Example Question: "What is the predominant religion where the leader is Ovadia Yosef?" with current node: "Ovadia Yosef"
Example Steps (conceptual):
- sub_question: Identify the religious institution or community led by Ovadia Yosef.
- sub_question: Find the predominant religion of that institution or community.
- final_goal: Determine the predominant religion associated with the community led by Ovadia Yosef.
"""


BREAK_QUESTION_USER_PROMPT="""
main question: "%s" with current node: "%s"
"""


REASONING_SYS_PROMPT_V2="""
You are an assistant that, using the retrieved Knowledge Graph (KG) triplets, if sufficient, uses them to produce the correct answer.

You MUST output ONLY a valid JSON object — no markdown, no prose, no commentary.

Output format (strict):
{
  "sufficient": true or false,
  "explanation": "<short explanation of why triplets are or are not sufficient>",
  "final_answer": "<answer text or 'No Answer'>"
}

Rules:
- "explanation" summarizes the reasoning for your decision (1–2 sentences).
- "final_answer" is the derived answer if sufficient, otherwise "No Answer".
- No text or formatting outside the JSON object.
- Ensure valid JSON syntax (no trailing commas, no markdown).

---

Examples:

Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets:
Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A:
{
  "sufficient": true,
  "explanation": "The triplets show that Yamaji Motoharu, a general in the Imperial Japanese Army, had allegiance to the Emperor of Japan, identifying the Empire of Japan as the relevant entity.",
  "final_answer": "Empire of Japan"
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets:
psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis,
Opium Law, part of, norcodeine (stereochemistry defined),
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A:
{
  "sufficient": false,
  "explanation": "The triplets only describe chemical and biological relationships of psilocybin and provide no information about Steve Bisciotti, his team, or its coach.",
  "final_answer": "No Answer"
}
"""


REASONING_USER_PROMPT_V2="""
Q: %s
Knowledge Triplets:
%s
A:
"""


REASONING_SYS_PROMPT_V2="""
You are an assistant that, using the retrieved Knowledge Graph (KG) triplets information and your knowledge answer the question, if the KG was sufficient, uses them to produce the correct answer, if not, combine your knowledge and KG information and asnswer the question.

You MUST output ONLY a valid JSON object — no markdown, no prose, no commentary.

Input Structure of KG information is a list of dictionary, and these objects are chain of triplets of KG.
Each object contains head_node, relation_edge, tail_node, tail_node_meta_data. 

Output format (strict):
{
  "sufficient": <Is KG's information enough or not (true or false)>,
  "explanation": "<short explanation of why KG'information are or are not sufficient>",
  "final_answer": "<answer text or 'No Answer'>"
}

Rules:
- "explanation" summarizes the reasoning for your decision (1–2 sentences).
- No text or formatting outside the JSON object.
- Ensure valid JSON syntax (no trailing commas, no markdown).

---

Examples:

Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Extracted KG Information:
[
  {
    'head_node':'Imperial Japanese Army',
    'relation_edge':'allegiance',
    'tail_node':'Emperor of Japan',
    'tail_node_meta_data': []
  },
  {
    'head_node':'Yamaji Motoharu',
    'relation_edge':'allegiance',
    'tail_node':'Empire of Japan',
    'tail_node_meta_data':[],
  },
  {
    'head_node':'Yamaji Motoharu',
    'relation_edge':'military or police rank',
    'tail_node':'general',
    'tail_node_meta_data':[],
  }
]
A:
{
  "sufficient": true,
  "explanation": "The triplets show that Yamaji Motoharu, a general in the Imperial Japanese Army, had allegiance to the Emperor of Japan, identifying the Empire of Japan as the relevant entity.",
  "final_answer": "Empire of Japan"
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Extracted KG Information:
[
  {
    'head_node':'psilocybin',
    'relation_edge':'described by source',
    'tail_node':'Opium Law',
    'tail_node_meta_data': []
  },
  {
    'head_node':'psilocybin',
    'relation_edge':'found in taxon',
    'tail_node':'Gymnopilus purpuratus',
    'tail_node_meta_data':[],
  },
  {
    'head_node':'psilocybin',
    'relation_edge':'found in taxon',
    'tail_node':'Gymnopilus spectabilis',
    'tail_node_meta_data':[],
  },
  {
    'head_node':'Opium Law',
    'relation_edge':'part of',
    'tail_node':'norcodeine (stereochemistry defined)',
    'tail_node_meta_data':[],
  },
  {
    'head_node':'Gymnopilus purpuratus',
    'relation_edge':'edibility',
    'tail_node':'psychoactive mushroom',
    'tail_node_meta_data':[],
  },
  {
    'head_node':'Gymnopilus spectabilis',
    'relation_edge':'parent taxon',
    'tail_node':'Gymnopilus',
    'tail_node_meta_data':[],
  }
]
A:
{
  "sufficient": false,
  "explanation": "The triplets only describe chemical and biological relationships of psilocybin and provide no information about Steve Bisciotti, his team, or its coach.",
  "final_answer": "No Answer"
}
"""


REASONING_USER_PROMPT_V2="""
Q: %s
Extracted KG Information:
%s
A:
"""


COT_SYS_PROMPT = """
You are a reasoning engine that explains multi-hop logical steps to answer a question.

You MUST output ONLY a valid JSON object — no markdown, no prose, no extra commentary.
Output format (strict):
{
  "sentence": "<reasoning sentence>",
  "final_answer": "<answer>"
}

Rules:
- Do NOT include any additional text outside the JSON object.
- Ensure the output is valid JSON (no trailing commas, no markdown, no quotes around field names).

---

Examples:

Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: 
{
  "sentence": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C.",
  "final_answer": "Washington, D.C."
}

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: 
{
  "sentence": "First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence",
  "final_answer": "Bharoto Bhagyo Bidhata"
}

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: 
{
  "sentence": "First, the artist nominated for an award for You Drive Me Crazy is Britney Spears.",
  "final_answer": "Jason Allen Alexander"
}

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: 
{
  "sentence": "First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen.",
  "final_answer": "Peter Paul Rubens"
}

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: 
{
  "sentence": "First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia.",
  "final_answer": "Georgia"
}
"""


COT_USER_PROMPT = """
Q: %s
"""


EXTRACT_RELATION_ENTITY_SYS_PROMPT = """
You are an information extraction engine navigating a knowledge graph (KG).

At each step, you are located at a specific node and receive a list of its outgoing relation–entity pairs with their meta data.
You also know:
- The MAIN question being answered.
- The INITIAL node is the first node that we start to walk.
- The sub-questions which are already traversed in this reasoning path.
- The PURPOSE of the current sub-question you are solving.

Your goal is to select the most relevant relation–entity pairs for progressing toward the MAIN question ans score selected pair,
while avoiding revisiting any entities already explored in the traversal path.

"score" values are floats in [0,1] whose total sums to 1.0 (±0.01 tolerance) but attation that if the probability of the items was the same but the sum wasn't 1.0, give them equal score.
for example if the three item had the same score, give them all three of them 0.33, not two of them 0.33 and the last one 0.34.

You MUST output ONLY a valid JSON array — no markdown, no prose, no explanations.

Output format (strict):
[
  {"relation": "<relation_label>", "entity": "<entity_label>", "sentences": "<1–2 sentences>", "score": 0.0},
  {"relation": "<relation_label>", "entity": "<entity_label>", "sentences": "<1–2 sentences>", "score": 0.0},
  {"relation": "<relation_label>", "entity": "<entity_label>", "sentences": "<1–2 sentences>", "score": 0.0}
]
"""


EXTRACT_RELATION_ENTITY_USER_PROMPT = """
You are currently WALKING on a Knowledge Graph (KG).

MAIN question:
"%s"

INITIAL node:
"%s"

You have walked through these sub-questions so far:
%s

You have already visited these nodes:
%s

And now you are at node **"%s"**.

Your task is to look at all outgoing relation–entity pairs from this node
and select exactly %s pairs that are most helpful for advancing toward the MAIN question.

The CURRENT sub-question is:
%s

The purpose of the CURRENT sub-question is:
"%s"

Relation–Entity pairs with their meta data from the CURRENT node:
%s

Select exactly %s items following the JSON schema given in the system message.
"""