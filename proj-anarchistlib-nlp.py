
'''
C. McClintock 
Anarchist Library - natural language processing
'''

import pandas as pd
import spacy
from glob import glob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import re 


# read in jsons
jsons = []
for f_name in glob('./data/*.json'): # for all jsons in directory
    with open(f_name, "r") as fh: 
        file = json.loads(fh.read())
        jsons.extend(file)

df = pd.DataFrame(jsons)

df = df.head(500)

#df['title'] = re.sub('\n', '', df['title'].str) # TO DO: fix titles

# how many documents to we have from each author?
# print(df['author'].value_counts())


tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_vector = tfidf_vectorizer.fit_transform(df['text'])

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=df['title'], columns=tfidf_vectorizer.get_feature_names_out())

# look at 
tfidf_slice = tfidf_df[['government', 'state']]
tfidf_slice.sort_index().round(decimals=2)

tfidf_df['doc_title'] = tfidf_df.index

tfidf_long = pd.melt(tfidf_df, id_vars='doc_title', value_vars=tfidf_df.columns[:-2], var_name='term', value_name='tfidf')

top10s = tfidf_long.sort_values(by=['doc_title','tfidf'], ascending=[True,False]).groupby(['doc_title']).head(10)

print(top10s)

top10s.to_csv('results/tfidf-test.csv')


'''
ideas: 
https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
sentiment analysis - look at 10 most + and -
NER - who appears most?
remove \n from tiles 
'''

