from app.prompt_list import *
import json
import time
from openai import OpenAI
import re
# from rank_bm25 import BM25Okapi
# from sentence_transformers import util
# from sentence_transformers import SentenceTransformer

# def retrieve_top_docs(query, docs, model, width=3):
#     """
#     Retrieve the topn most relevant documents for the given query.

#     Parameters:
#     - query (str): The input query.
#     - docs (list of str): The list of documents to search from.
#     - model_name (str): The name of the SentenceTransformer model to use.
#     - width (int): The number of top documents to return.

#     Returns:
#     - list of float: A list of scores for the topn documents.
#     - list of str: A list of the topn documents.
#     """

#     query_emb = model.encode(query)
#     doc_emb = model.encode(docs)

#     scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

#     doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

#     top_docs = [pair[0] for pair in doc_score_pairs[:width]]
#     top_scores = [pair[1] for pair in doc_score_pairs[:width]]

#     return top_docs, top_scores


# def compute_bm25_similarity(query, corpus, width=3):
#     """
#     Computes the BM25 similarity between a question and a list of relations,
#     and returns the topn relations with the highest similarity along with their scores.

#     Args:
#     - question (str): Input question.
#     - relations_list (list): List of relations.
#     - width (int): Number of top relations to return.

#     Returns:
#     - list, list: topn relations with the highest similarity and their respective scores.
#     """

#     tokenized_corpus = [doc.split(" ") for doc in corpus]
#     bm25 = BM25Okapi(tokenized_corpus)
#     tokenized_query = query.split(" ")

#     doc_scores = bm25.get_scores(tokenized_query)
    
#     relations = bm25.get_top_n(tokenized_query, corpus, n=width)
#     doc_scores = sorted(doc_scores, reverse=True)[:width]

#     return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


# def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
#     relations = []
#     if if_all_zero(topn_scores):
#         topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
#     i=0
#     for relation in topn_relations:
#         if relation in head_relations:
#             relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
#         else:
#             relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
#         i+=1
#     return True, relations


def run_llm(
    prompt: str,
    temperature: float,
    max_tokens: int,
    openai_api_keys: str,
    engine: str = "llama3",
    system_prompt: str = "You are an AI assistant that helps people find information.",
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    retries: int = 3,
    retry_delay: int = 2
) -> str:
    
    if "llama" in engine.lower():
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='EMPTY',
        )
    else:
        client = OpenAI(
            api_key=openai_api_keys,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)

    raise RuntimeError("All retry attempts failed.")

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_json(question, answer, walker, file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": walker.print_paths() if walker is not list else []}
    with open("ToG_{}.json".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def extract_answer(text):
    pattern = r"@(True|False)@\.\s*@(.*?)@(?:\.\s*Therefore, the answer to the question is @(.*?)@)?"

    match = re.match(pattern, text)
    if match:
        answered = match.group(1) if match.group(1) else False
        explanation = match.group(2)
        answer = match.group(3) if match.group(3) else None
        return {
            "answered":answered,
            "explanation": explanation,
            "answer": answer
        }
    else:
        return None
    


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args['temperature_reasoning'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('./data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string