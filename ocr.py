import itertools
import requests
import pdf2image
import pytesseract
import nltk
import hashlib
from neo4j import GraphDatabase
import pandas as pd
from transformers import AutoTokenizer
from zero_shot_re import RelTaggerModel, RelationExtractor


nltk.download("punkt")


def clean_text(text):
    """Remove section titles and figure descriptions from text"""
    clean = "\n".join(
        [
            row
            for row in text.split("\n")
            if (len(row.split(" "))) > 3
            and not (row.startswith("(a)"))
            and not row.startswith("Figure")
        ]
    )
    return clean


def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    """Biomedical entity linking API"""
    return requests.post(url, json={"text": str(text)}).json()


host = "bolt://54.160.35.51:7687"
user = "neo4j"
password = "poles-fuse-precautions"
driver = GraphDatabase.driver(host, auth=(user, password))



def neo4j_query(query, params=None):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())


pdf = requests.get("https://arxiv.org/pdf/2110.03526.pdf")
doc = pdf2image.convert_from_bytes(pdf.content)  # Get the article text
article = []

for page_number, page_data in enumerate(doc):
    txt = pytesseract.image_to_string(page_data).encode(
        "utf-8"
    )  # Sixth page are only references
    if page_number < 6:
        article.append(txt.decode("utf-8"))

article_txt = " ".join(article)


text = article_txt.split("INTRODUCTION")[1]
ctext = clean_text(text)
sentences = nltk.tokenize.sent_tokenize(ctext)


entity_list = []

# The last sentence is invalid
for s in sentences[:-1]:
    entity_list.append(query_plain(s))

parsed_entities = []
for entities in entity_list:
    e = []
    if not entities.get("annotations"):
        parsed_entities.append(
            {
                "text": entities["text"],
                "text_sha256": hashlib.sha256(
                    entities["text"].encode("utf-8")
                ).hexdigest(),
            }
        )
        continue
    for entity in entities["annotations"]:
        other_ids = [id for id in entity["id"] if not id.startswith("BERN")]
        entity_type = entity["obj"]
        entity_name = entities["text"][entity["span"]["begin"] : entity["span"]["end"]]
        try:
            entity_id = [id for id in entity["id"] if id.startswith("BERN")][0]
        except IndexError:
            entity_id = entity_name
            e.append(
                {
                    "entity_id": entity_id,
                    "other_ids": other_ids,
                    "entity_type": entity_type,
                    "entity": entity_name,
                }
            )

    parsed_entities.append(
        {
            "entities": e,
            "text": entities["text"],
            "text_sha256": hashlib.sha256(entities["text"].encode("utf-8")).hexdigest(),
        }
    )

author = article_txt.split("\n")[0]
title = " ".join(article_txt.split("\n")[2:4])
neo4j_query(
    """
MERGE (a:Author{name:$author})
MERGE (b:Article{title:$title})
MERGE (a)-[:WROTE]->(b)
""",
    {"title": title, "author": author},
)
# import the sentences and mentioned entities
neo4j_query("""
MATCH (a:Article)
UNWIND $data as row
MERGE (s:Sentence{id:row.text_sha256})
SET s.text = row.text
MERGE (a)-[:HAS_SENTENCE]->(s)
WITH s, row.entities as entities
UNWIND entities as entity
MERGE (e:Entity{id:entity.entity_id})
ON CREATE SET e.other_ids = entity.other_ids,
e.name = entity.entity,
e.type = entity.entity_type
MERGE (s)-[m:MENTIONS]->(e)
ON CREATE SET m.count = 1
ON MATCH SET m.count = m.count + 1
""", {'data': parsed_entities})

model = RelTaggerModel.from_pretrained("fractalego/fewrel-zero-shot")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
relations = ['associated', 'interacts']
extractor = RelationExtractor(model, tokenizer, relations)

# Candidate sentence where there is more than a single entity present
candidates = [s for s in parsed_entities if (s.get('entities')) and (len(s['entities']) > 1)]
predicted_rels = []
for c in candidates:
    combinations = itertools.combinations([{'name':x['entity'], 'id':x['entity_id']} for x in c['entities']], 2)
    for combination in list(combinations):
        try:
            ranked_rels = extractor.rank(text=c['text'].replace(",", " "), head=combination[0]['name'], tail=combination[1]['name'])
            # Define threshold for the most probable relation
            if ranked_rels[0][1] > 0.85:
                predicted_rels.append({'head': combination[0]['id'], 'tail': combination[1]['id'], 'type':ranked_rels[0][0], 'source': c['text_sha256']})
        except:
            pass


neo4j_query("""
UNWIND $data as row
MATCH (source:Entity {id: row.head})
MATCH (target:Entity {id: row.tail})
MATCH (text:Sentence {id: row.source})
MERGE (source)-[:REL]->(r:Relation {type: row.type})-[:REL]-
>(target)
MERGE (text)-[:MENTIONS]->(r)
""", {'data': predicted_rels})
# examine the extracted relationships
neo4j_query("""
MATCH (s:Entity)-[:REL]->(r:Relation)-[:REL]->(t:Entity), (r)<-
[:MENTIONS]-(st:Sentence)
RETURN s.name as source_entity, t.name as target_entity, r.type as
type, st.text as source_text
""")

pdf = requests.get("https://link.springer.com/content/pdf/10.1007/s00427-012-0426-4.pdf")
doc = pdf2image.convert_from_bytes(pdf.content)  # Get the article text
article = []

for page_number, page_data in enumerate(doc):
    txt = pytesseract.image_to_string(page_data).encode(
        "utf-8"
    )  # Sixth page are only references
    if page_number < 13:
        article.append(txt.decode("utf-8"))

article_txt = " ".join(article)



text = article_txt.split("Introduction")[1]
ctext = clean_text(text)
sentences = nltk.tokenize.sent_tokenize(ctext)

entity_list = []

# The last sentence is invalid
for s in sentences[:-1]:
    entity_list.append(query_plain(s))

parsed_entities = []
for entities in entity_list:
    e = []
    if not entities.get("annotations"):
        parsed_entities.append(
            {
                "text": entities["text"],
                "text_sha256": hashlib.sha256(
                    entities["text"].encode("utf-8")
                ).hexdigest(),
            }
        )
        continue
    for entity in entities["annotations"]:
        other_ids = [id for id in entity["id"] if not id.startswith("BERN")]
        entity_type = entity["obj"]
        entity_name = entities["text"][entity["span"]["begin"] : entity["span"]["end"]]
        try:
            entity_id = [id for id in entity["id"] if id.startswith("BERN")][0]
        except IndexError:
            entity_id = entity_name
            e.append(
                {
                    "entity_id": entity_id,
                    "other_ids": other_ids,
                    "entity_type": entity_type,
                    "entity": entity_name,
                }
            )

    parsed_entities.append(
        {
            "entities": e,
            "text": entities["text"],
            "text_sha256": hashlib.sha256(entities["text"].encode("utf-8")).hexdigest(),
        }
    )

lines = article_txt.split("\n")
author = lines[7].strip()
title = lines[5].strip()

neo4j_query(
    """
MERGE (a:Author{name:$author})
MERGE (b:Article{title:$title})
MERGE (a)-[:WROTE]->(b)
""",
    {"title": title, "author": author},
)
# import the sentences and mentioned entities
neo4j_query("""
MATCH (a:Article {title: $title})
UNWIND $data as row
MERGE (s:Sentence{id:row.text_sha256})
SET s.text = row.text
MERGE (a)-[:HAS_SENTENCE]->(s)  
WITH s, row.entities as entities
UNWIND entities as entity
MERGE (e:Entity{id:entity.entity_id})
ON CREATE SET e.other_ids = entity.other_ids,
e.name = entity.entity,
e.type = entity.entity_type
MERGE (s)-[m:MENTIONS]->(e)
ON CREATE SET m.count = 1
ON MATCH SET m.count = m.count + 1
""", {'title': title, 'data': parsed_entities})

# Candidate sentence where there is more than a single entity present
candidates = [s for s in parsed_entities if (s.get('entities')) and (len(s['entities']) > 1)]
predicted_rels = []
for c in candidates:
    combinations = itertools.combinations([{'name':x['entity'], 'id':x['entity_id']} for x in c['entities']], 2)
    for combination in list(combinations):
        try:
            ranked_rels = extractor.rank(text=c['text'].replace(",", " "), head=combination[0]['name'], tail=combination[1]['name'])
            # Define threshold for the most probable relation
            if ranked_rels[0][1] > 0.85:
                predicted_rels.append({'head': combination[0]['id'], 'tail': combination[1]['id'], 'type':ranked_rels[0][0], 'source': c['text_sha256']})
        except:
            pass


neo4j_query("""
UNWIND $data as row
MATCH (source:Entity {id: row.head})
MATCH (target:Entity {id: row.tail})
MATCH (text:Sentence {id: row.source})
MERGE (source)-[:REL]->(r:Relation {type: row.type})-[:REL]-
>(target)
MERGE (text)-[:MENTIONS]->(r)
""", {'data': predicted_rels})
# examine the extracted relationships
neo4j_query("""
MATCH (s:Entity)-[:REL]->(r:Relation)-[:REL]->(t:Entity), (r)<-
[:MENTIONS]-(st:Sentence)
RETURN s.name as source_entity, t.name as target_entity, r.type as
type, st.text as source_text
""")