from bs4 import BeautifulSoup
import os, json, time, requests
from typing import Dict, Any, List, Set
    
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
        

class WikiDataC():
    def __init__(self, lang: str = "en"):
        self.api = "https://www.wikidata.org/w/api.php"
        self.lang = lang
        self.database = './Graph/'
        os.makedirs(self.database, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WikiDataC/1.0 (contact: your@email)"})

    # ---------- low-level helpers ----------
    def _wbgetentities(self, ids: List[str], props: str) -> Dict[str, Any]:
        """Batch-call wbgetentities for up to 50 IDs."""
        out = {}
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
            data = r.json()
            out.update(data.get("entities", {}))
            time.sleep(0.05)  # be polite
        return out

    def _get_labels_for_ids(self, ids: Set[str]) -> Dict[str, str]:
        """Return {id: label} for any mixture of Q…/P… IDs. Missing labels -> id itself."""
        if not ids:
            return {}
        entities = self._wbgetentities(list(ids), props="labels")
        labels = {}
        for eid, e in entities.items():
            lbl = e.get("labels", {}).get(self.lang, {}).get("value")
            labels[eid] = lbl if lbl else eid
        return labels

    # ---------- public API ----------
    def get_entity_info(self, entity_qid: str) -> Dict[str, Any]:
        s = time.time()
        """Return label/description and claims with resolved property & value labels."""
        path = os.path.join(self.database, f"{entity_qid}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 1) Fetch the entity (labels, descriptions, claims)
        entity = self._wbgetentities([entity_qid], props="labels|descriptions|claims").get(entity_qid, {})

        label = entity.get("labels", {}).get(self.lang, {}).get("value", "")
        descr = entity.get("descriptions", {}).get(self.lang, {}).get("value", "")

        claims = entity.get("claims", {}) or {}

        # 2) Collect all property IDs and all target entity IDs used in mainsnaks
        prop_ids: Set[str] = set()
        value_entity_ids: Set[str] = set()

        for pid, stmt_list in claims.items():
            prop_ids.add(pid)
            for stmt in stmt_list:
                snak = stmt.get("mainsnak", {})
                datavalue = snak.get("datavalue", {})
                dv = datavalue.get("value")
                if snak.get("datatype") == "wikibase-item" and isinstance(dv, dict):
                    qid = dv.get("id")
                    if qid:
                        value_entity_ids.add(qid)

        # 3) Resolve labels (properties + target items) in batches
        id2label = self._get_labels_for_ids(prop_ids.union(value_entity_ids))

        # 4) Build your desired structure
        entity_info = {
            "label": label,
            "descriptions": descr,
            "claims": {}
        }

        for pid, stmt_list in claims.items():
            # property label
            p_label = id2label.get(pid, pid)

            entities_out = []
            for stmt in stmt_list:
                snak = stmt.get("mainsnak", {})
                datavalue = snak.get("datavalue")
                datatype = snak.get("datatype")

                if not datavalue:
                    continue

                # Handle different datatypes
                if datatype == "wikibase-item":
                    qid = datavalue["value"].get("id")
                    if qid:
                        entities_out.append({"label": id2label.get(qid, qid), "qid": qid})
                elif datatype in {"string", "external-id", "url"}:
                    entities_out.append({"label": str(datavalue["value"]), "qid": None})
                elif datatype == "monolingualtext":
                    v = datavalue["value"]
                    entities_out.append({"label": v.get("text", ""), "qid": None})
                elif datatype == "quantity":
                    v = datavalue["value"]
                    amount = v.get("amount")
                    unit = v.get("unit", "")
                    if unit.startswith("http"):
                        # unit often is another entity IRI like http://www.wikidata.org/entity/Q11573
                        unit_qid = unit.rsplit("/", 1)[-1]
                        entities_out.append({"label": f"{amount} {id2label.get(unit_qid, unit_qid)}", "qid": None})
                    else:
                        entities_out.append({"label": f"{amount}", "qid": None})
                elif datatype == "time":
                    v = datavalue["value"]
                    # ISO-like time (e.g., +1998-01-01T00:00:00Z)
                    entities_out.append({"label": v.get("time", ""), "qid": None})
                else:
                    # Fallback: stringify value
                    entities_out.append({"label": str(datavalue.get("value", "")), "qid": None})

            entity_info["claims"][pid] = {"label": p_label, "entities": entities_out}

        # 5) Cache
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entity_info, f, ensure_ascii=False, indent=2)

        e = time.time()
        process_time = e - s
        print(f"{entity_qid = }, Process time:", process_time, "seconds")
        return entity_info

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
        return entity_info["claims"].get(relation_pid, {}).get("entities", [])

class WikiDataC_v2():
    def __init__(self, lang: str = "en"):
        self.api = "https://www.wikidata.org/w/api.php"
        self.lang = lang
        self.database = './Graph/'
        os.makedirs(self.database, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WikiDataC/1.0 (contact: your@email)"})

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
            time.sleep(0.05)  # be polite
        return out

    def _get_meta_for_ids(self, ids: Set[str]) -> Dict[str, Dict[str, str]]:
        """Return {id: {'label':..., 'description':...}} for any mixture of Q…/P… IDs."""
        if not ids:
            return {}
        entities = self._wbgetentities(list(ids), props="labels|descriptions")
        out = {}
        for eid, e in entities.items():
            lbl = e.get("labels", {}).get(self.lang, {}).get("value")
            desc = e.get("descriptions", {}).get(self.lang, {}).get("value")
            out[eid] = {"label": lbl or eid, "description": desc or ""}
        return out

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


# collector = WikiDataC()
# q = 'Q9696'
# s = time.time()
# entity_info = collector.get_entity_info(q)
# e = time.time()
# process_time = e - s
# print("Process time:", process_time, "seconds")