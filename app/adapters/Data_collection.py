from bs4 import BeautifulSoup
import os, json, time, requests
from typing import Dict, Any, List, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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


class WikiDataC_v2():
    def __init__(self, lang: str = "en"):
        self.api = "https://www.wikidata.org/w/api.php"
        self.lang = lang
        self.database = './Graph/'
        os.makedirs(self.database, exist_ok=True)
        self._id_cache_path = os.path.join(self.database, "_id_label_cache.json")
        self._id_cache: Dict[str, Dict[str, str]] = self._load_id_cache()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ThesisKGQA/1.0 (contact: YOUR_REAL_EMAIL@domain.com; academic research project)"
        })

        retry = Retry(
            total=6,
            backoff_factor=2,                 # waits ~2s, 4s, 8s, 16s, 32s, 64s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],          # use method_whitelist=["GET"] on urllib3 < 1.26
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ---------- low-level helpers ----------
    def _wbgetentities(self, ids: List[str], props: str) -> Dict[str, Any]:
        out = {}
        if not ids:
            return out
        for i in range(0, len(ids), 50):
            chunk = ids[i:i+50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "languages": self.lang,
                "props": props,
                "format": "json"
            }
            r = self.session.get(self.api, params=params, timeout=30)
            r.raise_for_status()
            out.update(r.json().get("entities", {}))
            time.sleep(0.3)  # be polite
        return out

    def _load_id_cache(self) -> Dict[str, Dict[str, str]]:
        if os.path.exists(self._id_cache_path):
            try:
                with open(self._id_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_id_cache(self):
        tmp = self._id_cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._id_cache, f, ensure_ascii=False)
        os.replace(tmp, self._id_cache_path)

    def _get_meta_for_ids(self, ids: Set[str]) -> Dict[str, Dict[str, str]]:
        if not ids:
            return {}
        missing = [i for i in ids if i not in self._id_cache]
        if missing:
            entities = self._wbgetentities(missing, props="labels|descriptions")
            for eid, e in entities.items():
                lbl = e.get("labels", {}).get(self.lang, {}).get("value")
                desc = e.get("descriptions", {}).get(self.lang, {}).get("value")
                self._id_cache[eid] = {"label": lbl or eid, "description": desc or ""}
            for i in missing:
                self._id_cache.setdefault(i, {"label": i, "description": ""})  # deleted/missing items
            self._save_id_cache()
        return {i: self._id_cache[i] for i in ids}

    # ---------- public API ----------
    def get_entity_info(self, entity_qid: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Label/description + all claims; each claim has statements with qualifiers (labels+descriptions resolved)."""
        path = os.path.join(self.database, f"{entity_qid}.json")
        if (not force_refresh) and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 1) Fetch the entity
        entity = self._wbgetentities([entity_qid], props="labels|descriptions|claims").get(entity_qid, {}) or {}
        label = entity.get("labels", {}).get(self.lang, {}).get("value", "")
        descr = entity.get("descriptions", {}).get(self.lang, {}).get("value", "")
        claims = entity.get("claims", {}) or {}

        # 2) Collect properties + all linked items (mainsnaks, qualifiers, and quantity units)
        prop_ids: Set[str] = set()
        item_ids: Set[str] = set()

        def collect_from_snak(snak: Dict[str, Any]):
            if snak.get("snaktype", "value") != "value":
                return
            dt = snak.get("datatype")
            dv = (snak.get("datavalue") or {}).get("value")

            if dt == "wikibase-item" and isinstance(dv, dict):
                qid = dv.get("id")
                if qid:
                    item_ids.add(qid)

            elif dt == "quantity" and isinstance(dv, dict):
                unit = dv.get("unit", "")
                if isinstance(unit, str) and unit.startswith("http://www.wikidata.org/entity/"):
                    item_ids.add(unit.rsplit("/", 1)[-1])


        for pid, stmt_list in claims.items():
            prop_ids.add(pid)
            for stmt in stmt_list:
                collect_from_snak(stmt.get("mainsnak", {}))
                for qpid, qsnaks in (stmt.get("qualifiers") or {}).items():
                    prop_ids.add(qpid)
                    for q in qsnaks:
                        collect_from_snak(q)

        # 3) Resolve labels+descriptions for everything we reference
        id2meta = self._get_meta_for_ids(prop_ids.union(item_ids))

        def format_snak(snak: Dict[str, Any]) -> Dict[str, Any]:
            # Handle snaks without a concrete value
            snaktype = snak.get("snaktype", "value")
            if snaktype != "value":
                # common cases: "somevalue" or "novalue"
                return {"qid": None, "label": snaktype, "description": ""}

            dt = snak.get("datatype")
            dv = (snak.get("datavalue") or {}).get("value")

            # Defensive guard: if there's allegedly a value but it's None/str/etc.
            if dv is None:
                return {"qid": None, "label": "", "description": ""}

            if dt == "wikibase-item":
                if isinstance(dv, dict):
                    qid = dv.get("id")
                    if qid:
                        m = id2meta.get(qid, {"label": qid, "description": ""})
                        return {"qid": qid, "label": m["label"], "description": m["description"]}
                # fallback
                return {"qid": None, "label": str(dv), "description": ""}

            if dt in {"string", "external-id", "url"}:
                return {"qid": None, "label": str(dv), "description": ""}

            if dt == "monolingualtext":
                if isinstance(dv, dict):
                    return {"qid": None, "label": dv.get("text", ""), "description": ""}
                return {"qid": None, "label": str(dv), "description": ""}

            if dt == "quantity":
                if isinstance(dv, dict):
                    amount = dv.get("amount")
                    unit = dv.get("unit", "")
                    if isinstance(unit, str) and unit.startswith("http://www.wikidata.org/entity/"):
                        uqid = unit.rsplit("/", 1)[-1]
                        ulabel = id2meta.get(uqid, {"label": uqid})["label"]
                        return {"qid": None, "label": f"{amount} {ulabel}", "description": ""}
                    return {"qid": None, "label": f"{amount}", "description": ""}
                # fallback
                return {"qid": None, "label": str(dv), "description": ""}

            if dt == "time":
                if isinstance(dv, dict):
                    return {"qid": None, "label": dv.get("time", ""), "description": ""}
                return {"qid": None, "label": str(dv), "description": ""}

            # Unknown datatype -> stringify
            return {"qid": None, "label": str(dv), "description": ""}


        # 4) Build rich structure (with qualifiers) + legacy entities
        out = {"label": label, "descriptions": descr, "claims": {}}

        for pid, stmt_list in claims.items():
            pmeta = id2meta.get(pid, {"label": pid, "description": ""})

            statements = []
            legacy_entities = []

            for stmt in stmt_list:
                main_value = format_snak(stmt.get("mainsnak", {}))
                legacy_entities.append({"label": main_value["label"], "qid": main_value["qid"]})

                quals_out = []
                for qpid, qsnaks in (stmt.get("qualifiers") or {}).items():
                    qmeta = id2meta.get(qpid, {"label": qpid, "description": ""})
                    for q in qsnaks:
                        quals_out.append({
                            "pid": qpid,
                            "property_label": qmeta["label"],
                            "property_description": qmeta.get("description", ""),
                            "value": format_snak(q)
                        })

                statements.append({
                    "value": main_value,
                    "qualifiers": quals_out,
                    "rank": stmt.get("rank")
                })

            out["claims"][pid] = {
                "label": pmeta["label"],
                "description": pmeta.get("description", ""),
                "statements": statements,
                "entities": legacy_entities
            }

        # 5) Cache
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    def get_all_relations_of_an_entity(self, entity_qid: str) -> List[Dict[str, str]]:
        entity_info = self.get_entity_info(entity_qid)
        all_relations = []
        for pid, payload in entity_info["claims"].items():
            all_relations.append({"pid": pid, "label": payload["label"]})
        return all_relations

    def label2pid(self, entity_qid: str, label: str):
        entity_info = self.get_entity_info(entity_qid)
        for pid, payload in entity_info["claims"].items():
            if payload["label"] == label:
                return pid
        return False

    def qid2label(self, entity_qid: str) -> str:
        info = self._wbgetentities([entity_qid], props="labels").get(entity_qid, {})
        return info.get("labels", {}).get(self.lang, {}).get("value", "")

    def get_tail_entities_given_head_and_relation(self, entity_qid: str, relation_pid: str):
        entity_info = self.get_entity_info(entity_qid)
        payload = entity_info["claims"].get(relation_pid, {})
        tail_entities = []
        for st in payload["statements"]:
            tail_entity = {}
            heading = st["value"].get("label", "")
            qid = st["value"].get("qid", "")
            tail_entity['label'] = heading
            tail_entity['qid'] = qid
            meta_data = []
            for q in st["qualifiers"]:
                meta_data.append(
                    {
                        q['property_label']: q['value']['label']
                    }
                )
            tail_entity['meta_data'] = meta_data
            tail_entities.append(tail_entity)
            
        return tail_entities
    
    
# collector = WikiDataC()
# q = 'Q9696'
# s = time.time()
# entity_info = collector.get_entity_info(q)
# e = time.time()
# process_time = e - s
# print("Process time:", process_time, "seconds")