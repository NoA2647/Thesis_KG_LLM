from app.prompt_list import *
import json
import time
from openai import OpenAI
import re

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
    

def save_2_json(data_id, question, results, paths, file_name):
    dict = {"id": data_id, "question":question, "results": results, "reasoning_chains": paths}
    with open(f"ToG_{file_name}.json", "a") as outfile:
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
        id_string = 'ID'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        id_string = 'QuestionId'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        id_string = None
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
        id_string = None
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
        id_string = None
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        id_string = None
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
        id_string = None
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
        id_string = None
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
        id_string = 'ex_id'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string, id_string