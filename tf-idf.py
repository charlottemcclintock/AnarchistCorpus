
'''
C. McClintock 
Anarchist Library - playing around with tf-idf
'''

# %% IMPORT LIBRARIES & READ IN FILES

import pandas as pd
from pandas.io.formats.format import return_docstring
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

# how many documents to we have from each author?
df['author'].value_counts().to_csv('results/author-counts.csv')

#%%
# clean titles 
titles = []
for index, row in df.iterrows(): 
    new_title = " ".join(row['title'].split())
    titles.append(new_title)

df['title'] = titles

'''
LOOKING AT UNIQUE WORDS
'''

#%% TF-IDF

# create vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# fit to data
tfidf_vector = tfidf_vectorizer.fit_transform(df['text'])

# to data frame
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=df['title'], columns=tfidf_vectorizer.get_feature_names_out())

# look at top 10 words for each document by tf-idf
tfidf_df['doc_title'] = tfidf_df.index
tfidf_long = pd.melt(tfidf_df, id_vars='doc_title', value_vars=tfidf_df.columns[:-2], var_name='term', value_name='tfidf')
top10s = tfidf_long.sort_values(by=['doc_title','tfidf'], ascending=[True,False]).groupby(['doc_title']).head(10)

# print and write out to csv
print(top10s)
top10s.to_csv('results/tfidf-full.csv')


''''
BUILDING A RECOMMENDATION ENGINE with tf-idf 
what should I read next?
'''
# %% RECOMMEND 10 OTHER WORKS BASED ON TF-IDF

from sklearn.metrics.pairwise import linear_kernel

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['text'])

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate mapping between titles and index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of work that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar works
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    work_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar works
    return df['title'].iloc[work_indices]


# Generate recommendations 
print(get_recommendations('Flow of Water', cosine_sim, indices))

# %%
# let's see how it's working
recs = list(get_recommendations('Young Property', cosine_sim, indices))
recs.append('Young Property')
top10s_yp = top10s[top10s['doc_title'].isin(recs)]

top10s_yp.to_csv('results/recs-tfidf.csv')

# %%
