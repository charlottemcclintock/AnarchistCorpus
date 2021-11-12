

# %%

import pandas as pd
import spacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from glob import glob
import json


# read in jsons
jsons = []
for f_name in glob('./data/*.json'): # for all jsons in directory
    with open(f_name, "r") as fh: 
        file = json.loads(fh.read())
        jsons.extend(file)

df = pd.DataFrame(jsons)

# %% most common entities 

text_data = []

for index, row in df.iterrows():
    try:
        tokens = nlp(row['text'])
        text_data.append(tokens)
        print(f"Prepared {row['title']}")

    except: 
        print('Line preparation error.')


# %%

items = []
for text in text_data: 
    for x in text.ents: 
        items.append(x.text)

Counter(items).most_common(20)


# %% person entities
person_list = []
for text in text_data: 
    for ent in text.ents:
        if ent.label_ == 'PERSON':
            person_list.append(ent.text)
            
person_counts = Counter(person_list).most_common(20)
df_person = pd.DataFrame(person_counts, columns =['text', 'count'])

# %%
ner_lst = nlp.pipe_labels['ner']

print(len(ner_lst))
print(ner_lst)

#%% check many entities
ent_list = []
for text in text_data: 
    for ent in text.ents:
        if ent.label_ == 'WORK_OF_ART':
            ent_list.append(ent.text)
            
ent_counts = Counter(ent_list).most_common(20)
print(ent_counts)
# %%
