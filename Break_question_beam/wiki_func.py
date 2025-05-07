from prompt_list import *
import json
import openai
import re
import time
from utils import *


def transform_relation(relation):
    relation_without_prefix = relation.replace("wiki.relation.", "").replace("_", " ")
    return relation_without_prefix

def clean_sub_questions(result):
    sub_questions = re.findall(r'\**[Ss]tep \d+:\** (.*?\?)', result)
    return True, sub_questions

def clean_relations(string, entity_id):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        relation = transform_relation(relation)
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"

        relations.append({"entity": entity_id, "relation_label": relation, "score": score})
    if not relations:
        return False, "No relations found"
    return True, relations

def clean_te_r_ce_triplets(result, te_r_ce_list):
    result = result.replace("**", "")
    pattern = r'(?P<topic_entity_label>[\w_.]+)\s*--\s*(?P<relation_label>[\w_.]+)\s*--\s*(?P<entity_candidate_label>[\w_.]+):\s*(?P<sentence>.*?)\s*Score:\s*(?P<score>\d+\.\d+)'
    scored_triplets = []
    for match in re.finditer(pattern, result):
        topic_entity_label = match.group("topic_entity_label").strip()
        relation_label = match.group("relation_label").strip()
        entity_candidate_label = match.group("entity_candidate_label").strip()
        
        # Handle the score safely
        score_str = match.group("score")
        score = float(score_str) if score_str else 0.0
        for l in te_r_ce_list:
            if topic_entity_label == l['topic_entity_label'].replace(" ","_") and entity_candidate_label == l['entity_candidate_label'].replace(" ","_"):
                l['score'] = score
                scored_triplets.append(l)
                # use break bc it may create redundant if TE and CE be more that one (strange right! but it happend) 
                break
    return bool(scored_triplets), scored_triplets

# Clean the relation-entity results:
def clean_relation_entity_pairs(result, entity_id, entity_label, relation_entity_pairs):
    result = result.replace("**", "")
    pattern = r'(?P<relation_label>[\w_]+)\s*--\s*(?P<entity_candidate_label>[\w_ .]+):\s*(?P<sentence>.*?)\s*Score:\s*(?P<score>\d+\.\d+)'
    entity_candidate_id = None
    scored_pairs = []
    for match in re.finditer(pattern, result):
        relation_label = match.group("relation_label").strip()
        entity_candidate_label = match.group("entity_candidate_label").strip()
        
        # Handle the score safely
        score_str = match.group("score")
        score = float(score_str) if score_str else 0.0
        for l in relation_entity_pairs:
            for candidate_name, candidate_id in zip(l['entity_candidates_label'], l['entity_candidates_id']):
                if entity_candidate_label == candidate_name.replace(" ","_"):
                    entity_candidate_id = candidate_id
                    entity_candidate_label = candidate_name
        scored_pairs.append({"topic_entity_id": entity_id, "topic_entity_label": entity_label, "relation_label": relation_label, "entity_candidate_id":entity_candidate_id, "entity_candidate_label": entity_candidate_label, "score": score})
    return bool(scored_pairs), scored_pairs

def construct_te_r_ce_prune_prompt(question, te_r_ce_list):
    result = ""
    idx = 1
    for triplet in te_r_ce_list:
        if triplet['entity_candidate_id'] != None:
            result += f"{idx}. {triplet["topic_entity_label"].replace(' ', '_')} -- {triplet['relation_label'].replace(' ', '_')} -- {triplet['entity_candidate_label'].replace(' ', '_')}\n"
            idx += 1

    prompt_str = score_te_r_ce_prompt %(question, result) 
    
    return prompt_str

def construct_relation_entity_prune_prompt(question, entity_name, relation_entity_pairs, args):
    result = ""
    idx = 1
    for pair in relation_entity_pairs:
        for candidate_name, candidate_id in zip(pair['entity_candidates_label'], pair['entity_candidates_id']):
            if candidate_id != None:
                result += f"{idx}. {pair['relation_label'].replace(' ', '_')} -- {candidate_name.replace(' ', '_')}\n"
                idx += 1
            
    prompt_str = extract_relation_entity_prompt %(entity_name, max(3,args['width']), question, result) 
    
    return prompt_str

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt_wiki % (args['width'], args['width'])+question+'\nTopic Entity: '+entity_name+ '\nRelations:\n'+'\n'.join([f"{i}. wiki.relation.{item.replace(' ', '_')}" for i, item in enumerate(total_relations, start=1)])+'\nA:'

def construct_break_question_prompt(question, entity_name):
    return break_question_propmt2 % (entity_name, question, question, entity_name)


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    useless_relation_list = ["category's main topic", "topic\'s main category", "stack exchange site", 'main subject', 'country of citizenship', "commons category", "commons gallery", "country of origin", "country", "nationality"]
    if check_end_word(relation) or 'wikidata' in relation.lower() or 'wikimedia' in relation.lower() or relation.lower() in useless_relation_list:
        return True
    return False


def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def retrieve_relation_entity_pairs(entity_id, pre_relations, args, wiki_client, llm_logger, session_key):
    # Retrieve all relations for the current entity
    relations = wiki_client.get_all_relations_of_an_entity(entity_id)
    head_relations = [rel['label'] for rel in relations]
    
    if args['remove_unnecessary_rel']:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
    head_relations = list(set(head_relations) - set(pre_relations))
    
    total_relations = list(set(head_relations))
    total_relations.sort()

    # Create relation-entity pairs
    relation_entity_pairs = []
    for relation in total_relations:
        entity_candidates_id, entity_candidates_name = entity_search(entity_id, relation, wiki_client)
        if entity_candidates_name:
            relation_entity_pairs.append({
                'topic_entity_id': entity_id,
                'relation_label': relation,
                'entity_candidates_id': entity_candidates_id,
                'entity_candidates_label': entity_candidates_name
            })
    llm_logger.debug(f'Relation-Entity pair:\n{relation_entity_pairs}', extra={'session_key': session_key})
    return relation_entity_pairs

def relation_entity_search_prune(entity_id, entity_name, question, relation_entity_pairs, args, llm_logger, session_key):
    # Now we prompt the LLM to prune and score these relation-entity pairs
    prompt = construct_relation_entity_prune_prompt(question, entity_name, relation_entity_pairs, args)
    llm_logger.debug(f'Relation-Entity Search Prune Prompt:\n{prompt}', extra={'session_key': session_key})
    result = run_llm(prompt, args['temperature_exploration'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f'Relation-Entity Search Prune result:\n{result}', extra={'session_key': session_key})
    # Clean and extract relation-entity pairs with scores
    flag, scored_pairs = clean_relation_entity_pairs(result, entity_id, entity_name, relation_entity_pairs)
    return scored_pairs if flag else []


def break_question(question, entity_name, args, llm_logger, session_key):
    prompt = construct_break_question_prompt(question, entity_name)
    llm_logger.debug(f'break_question prompt:\n{prompt}', extra={'session_key': session_key})
    result = run_llm(prompt, args['temperature_exploration'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f'Raw results:\n{result}', extra={'session_key': session_key})
    flag, sub_questions = clean_sub_questions(result)
    llm_logger.debug(f'clean_sub_questions:\n{sub_questions}', extra={'session_key': session_key})
    if flag:
        return sub_questions
    else:
        return []

def relation_search_prune(entity_id, entity_name, pre_relations, question, args, wiki_client, llm_logger, session_key):
    relations = wiki_client.get_all_relations_of_an_entity(entity_id)
    head_relations = [rel['label'] for rel in relations]
    
    if args['remove_unnecessary_rel']:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
    head_relations = list(set(head_relations) - set(pre_relations))
    
    total_relations = list(set(head_relations))
    total_relations.sort()  # make sure the order in prompt is always equal
    llm_logger.debug(f'Relations extracted from wikipedia:\n{total_relations}', extra={'session_key': session_key})
    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
    llm_logger.debug(f'relation_search_prune prompt:\n{prompt}\n', extra={'session_key': session_key})
    result = run_llm(prompt, args['temperature_exploration'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f'Raw results:\n{result}', extra={'session_key': session_key})
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    

def del_all_unknown_entity(entity_candidates_id, entity_candidates_name):
    if len(entity_candidates_name) == 1 and entity_candidates_name[0] == "N/A":
        return entity_candidates_id, entity_candidates_name

    new_candidates_id = []
    new_candidates_name = []
    for i, candidate in enumerate(entity_candidates_name):
        if candidate != "N/A":
            new_candidates_id.append(entity_candidates_id[i])
            new_candidates_name.append(candidate)

    return new_candidates_id, new_candidates_name


def all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def entity_search(entity_qid, relation_label, wiki_client):
    rid = wiki_client.label2pid(entity_qid, relation_label)
    if not rid or rid == "Not Found!":
        return [], []
    
    # rid_str = rid.pop()

    entities = wiki_client.get_tail_entities_given_head_and_relation(entity_qid, rid)

    id_list = [item['qid'] for item in entities]
    name_list = [item['label'] if item['label'] != "N/A" else "Unname_Entity" for item in entities]
    
    return id_list, name_list

# replace of entity_score
# (current_TE_R_CE_list, sub_question, arges(width), llm_logger, session_key)
def te_r_ce_score(te_r_ce_list, sub_question, args, llm_logger, session_key):
    if len(te_r_ce_list) == 1:
        return te_r_ce_list

    # prompt = construct_entity_score_prompt(question, relation_label, entity_candidates)
    prompt = construct_te_r_ce_prune_prompt(sub_question, te_r_ce_list)
    
    llm_logger.debug(f'te_r_ce prune prompt:\n{prompt}', extra={'session_key': session_key})
    result = run_llm(prompt, args['temperature_exploration'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f'raw result for te_r_ce_score:\n{result}', extra={'session_key': session_key})
    flag, scored_pairs = clean_te_r_ce_triplets(result, te_r_ce_list)
    llm_logger.debug(f'clean_te_r_ce scores:\n{scored_pairs}', extra={'session_key': session_key})
    
    return flag, scored_pairs


def entity_score(question, entity_candidates_id, entity_candidates, score, relation_label, args, llm_logger, session_key):
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    prompt = construct_entity_score_prompt(question, relation_label, entity_candidates)
    llm_logger.debug(f'prompt entity:\n{prompt}', extra={'session_key': session_key})
    result = run_llm(prompt, args['temperature_exploration'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f'raw result for \'{relation_label}\':\n{result}', extra={'session_key': session_key})
    entity_scores = clean_scores(result, entity_candidates)
    llm_logger.debug(f'clean_entity_scores:\n{entity_scores}', extra={'session_key': session_key})
    if all_zero(entity_scores):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    else:
        return [float(x) * score for x in entity_scores], entity_candidates, entity_candidates_id


def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag):
    if value_flag:
        scores = [1/len(entity_candidates) * entity['score']]
    candidates_relation = [entity['relation_label']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [1] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, walker, depth, args, llm_logger, session_key):
    llm_logger.debug("No new knowledge added during search depth %d, stop searching." % depth, extra={'session_key': session_key})
    answer = generate_answer(question, walker, args, llm_logger, session_key)
    save_2_json(question, answer, walker, file_name=args['dataset'])


def generate_answer(question, walker, args, llm_logger, session_key): 
    prompt = answer_prompt_wiki + question + '\n'
    prompt += "\nKnowledge Triplets: " + walker.format_for_llm() + 'A: '
    llm_logger.debug(f"generate haf stop answer prompt:\n{prompt}", extra={'session_key': session_key})
    result = run_llm(prompt, args['temperature_reasoning'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f"Final answer:{result}", extra={'session_key': session_key})
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, topic_entities_label, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, topic_entities_label, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[-1], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_topic_entities_label, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped], [x[6] for x in sorted_zipped]

    entities_id, relations, candidates, topics, topics_label, heads, scores = sorted_entities_id[:args['width']], sorted_relations[:args['width']], sorted_candidates[:args['width']], sorted_topic_entities[:args['width']], sorted_topic_entities_label[:args['width']], sorted_head[:args['width']], sorted_scores[:args['width']]
    merged_list = list(zip(entities_id, relations, candidates, topics, topics_label, heads, scores))
    filtered_list = [(id, rel, ent, top, topl, hea, score) for id, rel, ent, top, topl, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, topls, heads, scores = map(list, zip(*filtered_list))
    cluster_chain_of_entities = [(topls[i], relations[i], candidates[i], scores[i]) for i in range(len(candidates))]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, walker, args, llm_logger, session_key):
    prompt = prompt_evaluate_wiki + question
    prompt += "\nKnowledge Triplets: " + walker.format_for_llm() + 'A: '
    llm_logger.debug(f'reasoning prompt:\n{prompt}', extra={'session_key': session_key})
    response = run_llm(prompt, args['temperature_reasoning'], args['max_length'], args['opeani_api_keys'], args['LLM_type'])
    llm_logger.debug(f'raw result:\n{response}', extra={'session_key': session_key})
    result = extract_answer(response)
    llm_logger.debug(f'extract_answer:\n{result}', extra={'session_key': session_key})
    if if_true(result):
        return True, response
    else:
        return False, response
    

    