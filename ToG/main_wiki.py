from tqdm import tqdm
import random
from wiki_func import *
from Data_collection import *
from utils import *


if __name__ == '__main__':
    
    args = {'dataset' : 'cwq',
            'max_length' : 256, # the max length of LLMs output
            'temperature_exploration' : 0, # the temperature in exploration stage
            'temperature_reasoning' : 0, # the temperature in reasoning stage
            'width' : 3, # choose the search width of ToG
            'depth' : 3, # choose the search depth of ToG
            'remove_unnecessary_rel' : True, # whether removing unnecessary relations
            'LLM_type' : 'llama3.1', # LLM Type
            'opeani_api_keys' : "", # if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys
            'num_retain_entity' : 5, # Number of entities retained during entities search
            'prune_tools' : 'llm'}
    
    datas, question_string = prepare_dataset(args['dataset'])
    print(f"Start Running ToG on {args['dataset']} dataset.")
    for data in tqdm(datas[0:4]):
        question = data[question_string]
        print(f'Question:\n{question}\n')
        topic_entity = data['qid_topic_entity']
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            ## need change
            results = generate_without_explored_paths(question, args) 
            save_2_json(question, results, [], file_name=args['dataset'])
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        wiki_client = WikiDataB()
        for depth in range(1, args['depth']+1):
            print(f'\ndepth:{depth}')
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    print(f'Relation Prune ...')
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args, wiki_client)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                    print(f'Relations:\n{current_entity_relations_list}')
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                value_flag=False
                temp = entity['relation_label']
                print(f'Search [{temp}] ...')
                entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation_label'], wiki_client)
                print(f'Candidate Entity:\nf{entity_candidates_name}')
                if len(entity_candidates_name) == 0:
                    continue
                if len(entity_candidates_id) == 0: # values
                    value_flag=True
                    if len(entity_candidates_name) >= 20:
                        ## what is this?
                        entity_candidates_name = random.sample(entity_candidates_name, 10)
                    entity_candidates_id = ["[FINISH_ID]"] * len(entity_candidates_name)
                else: # ids
                    entity_candidates_id, entity_candidates_name = del_all_unknown_entity(entity_candidates_id, entity_candidates_name)
                    if len(entity_candidates_id) >=20:
                        indices = random.sample(range(len(entity_candidates_name)), 10)
                        entity_candidates_id = [entity_candidates_id[i] for i in indices]
                        entity_candidates_name = [entity_candidates_name[i] for i in indices]

                if len(entity_candidates_id) == 0:
                    continue
                print('Entity Score ...')
                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity_candidates_name, entity['score'], entity['relation_label'], args)
                print(f'Scores: {scores}')
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag)
            
            if len(total_candidates) == 0:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break
            print('Entity Prune ...')
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client)
            cluster_chain_of_entities.append(chain_of_entities)
            print(f'cluster chain of entities:\n{cluster_chain_of_entities}')
            if flag:
                print('Reasoning ...')
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                print(f'Stop:\n{stop}\nReasoning:\n{results}')
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_json(question, results, cluster_chain_of_entities, file_name=args['dataset'])
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.qid2label(entity) for entity in entities_id])}
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_json(question, results, [], file_name=args['dataset'])
