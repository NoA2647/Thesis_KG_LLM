from app.prompt_list import *
import json
import time
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import re
from typing import Optional, Dict, Any


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
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    openai_api_keys: str,
    sys_prompt: str = "You are an AI assistant that helps people find information.",
    engine: str = "llama3",
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

    messages: list[ChatCompletionMessageParam]= [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
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
            return response.choices[0].message.content or ""
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
    payload = {
        "id": data_id,
        "question": question,
        "answer": results,
        "paths": [p.to_dict() for p in paths],
    }
    with open(f"ToG_{file_name}.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_answer(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Be permissive about whitespace and case; explanation can be multiline.
    # Pattern captures:
    #  1) True/False
    #  2) Explanation (non-greedy, between @...@)
    #  3) Answer (optional group, between @...@ after the "Therefore..." clause)
    pattern = (
        r"@\s*(True|False)\s*@\.\s*@"          # @True@. or @False@.
        r"(?P<exp>.*?)"                        # explanation
        r"@\s*\.\s*Therefore,\s*the\s*answer\s*to\s*the\s*question\s*is\s*@"
        r"(?P<ans>.*?)"                        # answer
        r"@\s*\.?"                             # closing @ and optional trailing period
    )

    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        # Fallback: handle cases where the model omitted the final clause
        fallback_pattern = r"@\s*(True|False)\s*@\.\s*@(?P<exp>.*?)@"
        m2 = re.search(fallback_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not m2:
            return None
        answered_raw = m2.group(1).strip().lower()
        explanation = m2.group("exp").strip()
        answered = True if answered_raw == "true" else False
        # No explicit answer found
        return {
            "answered": answered,
            "explanation": explanation,
            "answer": None
        }

    answered_raw = m.group(1).strip().lower()
    explanation = m.group("exp").strip()
    answer_raw = (m.group("ans") or "").strip()

    answered = True if answered_raw == "true" else False

    answer = answer_raw

    return {
        "answered": answered,
        "explanation": explanation,
        "answer": answer
    }
    

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