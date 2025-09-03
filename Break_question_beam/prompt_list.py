break_question_propmt_v1 = """
break this question into multi-hop questions, first i give you an example then the question.
Example Question: 
"What is the predominant religion where the leader is Ovadia Yosef?" with initial node: "Ovadia Yosef".
Example Answer:
"Step 1: Who is Ovadia Yosef?

(Identify who Ovadia Yosef is, such as his background, role, and significance in a particular community.)

Step 2: What religious or spiritual role did Ovadia Yosef hold?

(Determine Ovadia Yosef's religious position, such as whether he was a religious leader, scholar, or figure in a specific faith.)

Step 3: What is the predominant religion associated with Ovadia Yosef's leadership?

(Identify the religion in which Ovadia Yosef was a leader, determining the community or faith group he influenced.)

By first identifying Ovadia Yosef's background, you can trace his religious significance and then identify the predominant religion associated with his leadership."

Now the main question: 
"%s" with initial node: "%s"
"""

# check model dont generate useless question (especially first question)
break_question_propmt_v2='''
I have a knowledge graph and I have the current node "%s" and I already know the information about this node so don't ask about this node. Break down the question "%s" into multi-hop sub-questions, starting from the current node.

Here’s an example of the expected output format:

Example Question:
"What is the predominant religion where the leader is Ovadia Yosef?" with current node: "Ovadia Yosef"

Example Answer:
Step 1: What religious or spiritual role did Ovadia Yosef hold?
(Determine Ovadia Yosef's religious position, such as whether he was a religious leader, scholar, or figure in a specific faith.)

Step 2: What is the predominant religion associated with Ovadia Yosef's leadership?
(Identify the religion in which Ovadia Yosef was a leader, determining the community or faith group he influenced.)

Now, for the main question:
"%s" with current node: "%s"
'''

score_te_r_ce_prompt = """
I'm walking on a knowledge graph with the following entity-relation-entity triplets. Please score all of these triplets base on contributation to answering the question "%s", in the following format <entity> -- <relation> -- <entity>: <sentences> Score:<score>. The sum of the scores should be 1.
Entity-Relation-Entity triplets:
%s
"""

# current_node, retrival_number, question, realtion-entity pairs 
extract_relation_entity_prompt = """
I have a knowledge graph with node "%s" and the following relation-entity pairs. Please retrieve %s of the relations that contribute to answering the question "%s", and provide the answer in the format <relation> -- <entity>: <sentences> Score:<score>. The sum of the scores of the relevant relations should be 1.
Relation-Entity Pairs:
%s
"""

# give correct outpute format for evaluation (@Yes/No@ and @Answer@)
reasoning_prompt="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge. give output in this format:
@True/False@. @Explanation@. Therefore, the answer to the question is @Answer@
for example:
Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: @True@. @Based on the given knowledge triplets and my knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan@. Therefore, the answer to the question is @Empire of Japan@.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis, 
Opium Law, part of, norcodeine (stereochemistry defined), 
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A: @False@. @Based on the given knowledge triplets and my knowledge, the specific psychedelic compound found in the Psilocybin genus mushroom that is converted to psilocin by the body is not explicitly mentioned@.
now the question is:

"""

half_stop_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your own knowledge.
@Explanation@. Therefore, the answer to the question is @Answer/need_information@
for example:
Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: Based on the given knowledge triplets and my knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis, 
Opium Law, part of, norcodeine (stereochemistry defined), 
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A: Based on the given knowledge triplets and my knowledge, the specific psychedelic compound found in the Psilocybin genus mushroom that is converted to psilocin by the body is not explicitly mentioned. Therefore, additional knowledge about the specific compounds and their conversion to psilocin is required to answer the question.

Q: """

cot_prompt = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}."""
