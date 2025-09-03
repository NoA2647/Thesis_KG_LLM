import requests
from bs4 import BeautifulSoup
import time
import json
import os
    
class WikiDataB():
    def __init__(self):
        self.root = "https://www.wikidata.org/wiki/"
        self.database = './Graph/'
        os.makedirs(self.database, exist_ok=True)

    def get_entity_info(self, entity_qid:str):
        path = self.database + entity_qid+'.json'
        if os.path.exists(path):
            with open(path, 'r') as file:
                entity_info = json.load(file)
                return entity_info
            
        else:
            url = self.root + entity_qid
            with requests.Session() as session:
                session.trust_env = False  # ignores HTTP_PROXY/HTTPS_PROXY from environment
                response = session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')  
            entity_info = {}
            
            entity_info['label'] = soup.find('span', class_='wikibase-title-label').text
            entity_info['descriptions'] = soup.find('div', class_='wikibase-entitytermsview-heading-description').text
            relations = soup.find_all('div', class_='wikibase-statementgroupview')
            entity_info['claims'] = {}
            for r in relations:
                r_code = r.get('id')
                # print(r_code)
                first_div = r.find('div', class_='wikibase-statementgroupview-property-label')
                r_label = first_div.find('a').text
                # print(r_label)
                # value = div class=wikibase-snakview-value wikibase-snakview-variation-valuesnak -> a code:title, label:text
                entities = []
                all_entities = r.find_all('div', class_='wikibase-statementview')
                for e in all_entities:
                    temp = e.find('div', class_='wikibase-snakview-value wikibase-snakview-variation-valuesnak')
                    t = temp.find('a')
                    # print(temp)
                    if t is not None:
                        e_label = t.text
                        e_code = t.get('title')
                    else:
                        e_label = temp.text
                        e_code = None
                    if not (e_label == "" and e_code == None):
                        entities.append({'label': e_label, 'qid': e_code})
                entity_info['claims'][r_code] = {'label':r_label, 'entities':entities}
                # print(entities)
            with open(f'{self.database}{entity_qid}.json', 'w') as f:
                    json.dump(entity_info, f)
            return entity_info
    
    def get_all_relations_of_an_entity(self, entity_qid: str):
        entity_info = self.get_entity_info(entity_qid)
        all_relations = []
        for r in entity_info['claims']:
            all_relations.append({'pid':r, 'label':entity_info['claims'][r]['label']})
            
        return all_relations

    def label2pid(self, entity_qid: str, label: str):
        entity_info = self.get_entity_info(entity_qid)
        for r in entity_info['claims']:
            if entity_info['claims'][r]['label'] == label:
                return r
        return False

    def qid2label(self, entity_qid: str):
        return self.get_entity_info(entity_qid)['label']
    
    def get_tail_entities_given_head_and_relation(self, entity_qid: str, relation_pid: str):
        entity_info = self.get_entity_info(entity_qid)
        return entity_info['claims'][relation_pid]['entities']
        
           
# collector = WikiDataB()
# q = 'Q31'
# s = time.time()
# entity_info = collector.get_entity_info(q)
# e = time.time()
# process_time = e - s
# print("Process time:", process_time, "seconds")