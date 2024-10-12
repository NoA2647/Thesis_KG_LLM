import logging
from tqdm import tqdm
import random
from wiki_func import *
from Data_collection import *
from utils import *
import uuid

if __name__ == '__main__':
    
    main_logger = logging.getLogger('main')
    main_handler = logging.FileHandler('Break_question_beam/main_log.log', encoding='utf-8')
    main_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | SessionKey: %(session_key)s | %(message)s')
    main_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_handler)
    main_logger.setLevel(logging.DEBUG)

    # Configure the second logger (for 'second_log.log')
    llm_logger = logging.getLogger('llm')
    llm_handler = logging.FileHandler('Break_question_beam/llm_log.log', encoding='utf-8')
    llm_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | SessionKey: %(session_key)s | %(message)s')
    llm_handler.setFormatter(llm_formatter)
    llm_logger.addHandler(llm_handler)
    llm_logger.setLevel(logging.DEBUG)
    
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
    for data in tqdm([datas[2], datas[19]]): # 19&2
        session_key = str(uuid.uuid4())
        question = data[question_string]
        main_logger.debug(f'Main Question:\n{question}', extra={'session_key': session_key})
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
        paths = []
        for entity in topic_entity:
            sub_questions = break_question(question, topic_entity[entity], args, llm_logger, session_key)
            temp = []
            for sub_question in sub_questions:
                temp.append({'current_ent_qid':entity, 
                             'current_ent_label':topic_entity[entity], 
                             'current_sub_question': sub_question})
            paths.append(temp)
        main_logger.debug(f'Paths:\n{paths}', extra={'session_key': session_key})
        for path in paths:
            if len(path) == 0:
                continue
        # while(step < args['depth'] and step < len(paths)-1):
            for step, stage in enumerate(path):
                main_logger.debug(f'step:{step}', extra={'session_key': session_key})
                current_entity_relations_list = []
                sub_question = stage['current_sub_question']
                entity_qid = stage['current_ent_qid']
                entity_label = stage['current_ent_label']
                if entity_qid!="[FINISH_ID]":
                    main_logger.debug(f'Relation Prune ...', extra={'session_key': session_key})
                    retrieve_relations_with_scores = relation_entity_search_prune(entity_qid, entity_label, pre_relations, sub_question, args, wiki_client, llm_logger, session_key)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                    main_logger.debug(f'Relations:\n{current_entity_relations_list}', extra={'session_key': session_key})
                total_candidates = []
                total_scores = []
                total_relations = []
                total_entities_id = []
                total_topic_entities = []
                total_head = []

                # new 
                if not current_entity_relations_list:
                    half_stop(question, cluster_chain_of_entities, step, args, llm_logger, session_key)
                    flag_printed = True
                    break
    
                # for entity in current_entity_relations_list:
                #     value_flag=False
                    # temp = entity['relation_label']
                    # main_logger.debug(f'Search Entities for [{temp}]...', extra={'session_key': session_key})
                    # entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation_label'], wiki_client)
                    # main_logger.debug(f'Candidate Entity:\nf{entity_candidates_name}', extra={'session_key': session_key})
                    # if len(entity_candidates_name) == 0:
                    #     continue
                    # if len(entity_candidates_id) == 0: # values
                    #     value_flag=True
                    #     if len(entity_candidates_name) >= 20:
                    #         entity_candidates_name = random.sample(entity_candidates_name, 10)
                    #     entity_candidates_id = ["[FINISH_ID]"] * len(entity_candidates_name)
                    # else: # ids
                    #     entity_candidates_id, entity_candidates_name = del_all_unknown_entity(entity_candidates_id, entity_candidates_name)
                    #     if len(entity_candidates_id) >=20:
                    #         indices = random.sample(range(len(entity_candidates_name)), 10)
                    #         entity_candidates_id = [entity_candidates_id[i] for i in indices]
                    #         entity_candidates_name = [entity_candidates_name[i] for i in indices]

                    # if len(entity_candidates_id) == 0:
                    #     continue
                    # main_logger.debug('Entity Score ...', extra={'session_key': session_key})
                    # scores, entity_candidates, entity_candidates_id = entity_score(sub_question, entity_candidates_id, entity_candidates_name, entity['score'], entity['relation_label'], args, llm_logger, session_key)
                    # main_logger.debug(f'Scores: {scores}', extra={'session_key': session_key})
                    # total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag)
            
                if len(total_candidates) == 0:
                    half_stop(question, cluster_chain_of_entities, step, args, llm_logger, session_key)
                    flag_printed = True
                    break
                
                main_logger.debug('Entity Prune ...', extra={'session_key': session_key})
                flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(
                    [item['entity'] for item in current_entity_relations_list],
                    [item['relation'] for item in current_entity_relations_list],
                    [item['entity'] for item in current_entity_relations_list],
                    [item['entity'] for item in current_entity_relations_list],
                    [1] * len(current_entity_relations_list),  # Assuming head = 1 for all
                    [item['score'] for item in current_entity_relations_list],
                    args,
                    wiki_client
                )

                cluster_chain_of_entities.append(chain_of_entities)
                main_logger.debug(f'cluster chain of entities:\n{cluster_chain_of_entities}', extra={'session_key': session_key})
                next_topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.qid2label(entity) for entity in entities_id])}
                if step+1 < len(path):
                    path[step+1]['current_ent_qid'] = list(next_topic_entity.keys())[0]
                    path[step+1]['current_ent_label'] = list(next_topic_entity.values())[0]
                if flag:
                    pass
                    # print('Reasoning ...')
                    # stop, results = reasoning(question, cluster_chain_of_entities, args)
                    # print(f'Stop:\n{stop}\nReasoning:\n{results}')
                    # if stop:
                    #     print("ToG stoped at depth %d." % step)
                    #     save_2_json(question, results, cluster_chain_of_entities, file_name=args['dataset'])
                    #     flag_printed = True
                    #     break
                    # else:
                    #     print("depth %d still not find the answer." % step)
                    #     flag_finish, entities_id = if_finish_list(entities_id)
                    #     if flag_finish:
                    #         half_stop(question, cluster_chain_of_entities, step, args)
                    #         flag_printed = True
                    #     else:
                    #         topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.qid2label(entity) for entity in entities_id])}
                    #         continue
                else:
                    half_stop(question, cluster_chain_of_entities, step, args, llm_logger, session_key)
                    flag_printed = True
                    
            main_logger.debug('Reasoning ...', extra={'session_key': session_key})
            stop, results = reasoning(question, cluster_chain_of_entities, args, llm_logger, session_key)
            main_logger.debug(f'Stop:\n{stop}\nReasoning:\n{results}', extra={'session_key': session_key})
            if stop:
                main_logger.debug("ToG stoped", extra={'session_key': session_key})
                save_2_json(question, results, cluster_chain_of_entities, file_name=args['dataset'])
                flag_printed = True
                break
            else:
                # need change when path ends it need to generate with cluster chain of entities
                main_logger.debug("ToG not find the answer in current path.", extra={'session_key': session_key})
                continue
                # flag_finish, entities_id = if_finish_list(entities_id)
                # if flag_finish:
                #     half_stop(question, cluster_chain_of_entities, step, args, llm_logger, session_key)
                #     flag_printed = True
                # else:
                    # need change
                    # topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.qid2label(entity) for entity in entities_id])}
                    # continue
                
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_json(question, results, [], file_name=args['dataset'])
