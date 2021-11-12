
'''
TOPIC MODELING WITH LATENT DIRICHLET ALLOCAITON
'''
# %%
import pandas as pd
import json
from glob import glob
import spacy
import random
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim import corpora, models
import pickle

#nltk.download('wordnet')

'''
STEP ONE: TOKENIZE TEXT AND PREPARE CORPUS
'''
#%%

# load spacy model
nlp = spacy.load('en_core_web_sm')

# define a tokenization function
def tokenize(text):
    lda_tokens = []
    tokens = nlp(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# get lemmas
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# get stop words in english
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

# set up text preparation with tokenize and lemmatize functions
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

# get jsons of documents
jsons = []
for f_name in glob('./data/*.json'): # for all jsons in directory
    with open(f_name, "r") as fh: 
        file = json.loads(fh.read())
        jsons.extend(file)

df = pd.DataFrame(jsons)

# prepare text 
text_data = []

for index, row in df.iterrows():
    try:
        tokens = prepare_text_for_lda(row['text'])
        text_data.append(tokens)
        print(f"Prepared {row['title']}")
    except: 
        print('Line preparation error.')


# create corpus
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

# save corpus and dictionary
pickle.dump(corpus, open('lda/corpus.pkl', 'wb'))
dictionary.save('lda/dictionary.gensim')


'''
RUN LDA!!
'''

# %% RUN MODELS
import gensim
from gensim import corpora, models
import pickle

# load dictionary and corpus
dictionary = gensim.corpora.Dictionary.load('lda/dictionary.gensim')
corpus = pickle.load(open('lda/corpus.pkl', 'rb'))

# run model
NUM_TOPICS = 20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('lda/model20.gensim')
topics = ldamodel.print_topics(num_words=6)
for topic in topics:
    print(topic)


#%% CREATE VISUALIZATION
lda = gensim.models.ldamodel.LdaModel.load('lda/model20.gensim')
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)

pyLDAvis.save_html(lda_display, 'results/lda-20.html')

#%% SAVE TOPIC WORDS
top_words_per_topic = []
for t in range(ldamodel.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in ldamodel.show_topic(t, topn = 20)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("results/lda_top_words.csv")


'''
Thanks to:
https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
'''

# %%
