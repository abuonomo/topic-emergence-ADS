import argparse
import logging
from pathlib import Path

import RAKE
import numpy as np
import pandas as pd
import spacy
import yake
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from mrakun import RakunDetector
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import roc_auc_score
from spacy.lang.en.stop_words import STOP_WORDS
from textacy.ke import textrank
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")
STEMMER = PorterStemmer()
TOPICS = ['x-ray', 'galax', 'star']


def load_data(infile, size=10_000):
    chunks = pd.read_json(infile, orient='records', lines=True, chunksize=size)
    dfc = next(chunks)

    topic_inds = {}
    for t in TOPICS:
        b_ind = dfc[~dfc['keyword'].isna()]['keyword'].apply(
            lambda x: sum([True if t in s.lower() else False for s in x]) > 0)
        ind = b_ind.index[b_ind].tolist()
        topic_inds[t] = ind

    df = pd.DataFrame(dfc['title'] + '. ' + dfc['abstract']).copy()
    df['keyword'] = dfc['keyword']
    df.columns = ['text', 'keyword']

    for t, ind in topic_inds.items():
        df[t] = np.zeros(df.shape[0])
        df.loc[ind, t] = 1
    df = df[df.iloc[:, 1:].sum(axis=1) > 0]
    LOG.info(f'Loading dataframe with shape: {df.shape}')
    return df


def get_extractors():
    hyperparameters = {
        "distance_threshold": 2,
        "distance_method": "editdistance",
        "num_keywords": 10,
        "pair_diff_length": 2,
        "stopwords": STOP_WORDS,
        "bigram_count_threshold": 2,
        "num_tokens": [1, 2],
        "max_similar": 3,  ## n most similar can show up n times
        "max_occurrence": 3  ## maximum frequency overall
    }
    rake_extractor = RAKE.Rake(list(STOP_WORDS))
    yake_extractor = yake.KeywordExtractor()
    rakun_extractor = RakunDetector(hyperparameters)

    return rake_extractor, yake_extractor, rakun_extractor


def get_textrank_kwds(df):
    LOG.info("Getting textrank keywords")
    df = df.copy()
    all_kwds = []
    for doc in tqdm(NLP.pipe(df['text'], batch_size=10), total=df.shape[0]):
        kwds = textrank(doc, normalize="lemma", topn=999)
        all_kwds.append(kwds)
    all_kwds_no_vals = [[k[0] for k in kwds] for kwds in all_kwds]
    df['textrank'] = all_kwds_no_vals
    return df


def get_rake_kwds(df):
    LOG.info("Getting rake keywords")
    df = df.copy()
    rake_extractor = RAKE.Rake(list(STOP_WORDS))
    f = lambda x: rake_extractor.run(x, minFrequency=1, minCharacters=3)
    rake_kwds = df['text'].progress_apply(f)
    rake_kwds_no_vals = [[k[0] for k in kwds] for kwds in rake_kwds]
    df['rake'] = rake_kwds_no_vals
    return df


def train_lda(kwds, n_topics=3):
    LOG.info("Training LDA")
    dct = Dictionary(kwds)
    corpus = [dct.doc2bow(t) for t in kwds]
    lda = LdaMulticore(corpus, n_topics, dct, passes=5, eval_every=1)
    cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    coh = cm.get_coherence()
    LOG.info(f'Coherence: {coh}')
    embedding = [lda.get_document_topics(c, minimum_probability=0) for c in corpus]
    embedding = np.vstack([np.array([e[1] for e in s]) for s in embedding])
    return lda, embedding, coh


def get_scores(df, model_dict, strats):
    records = []
    for t in TOPICS:
        for s in strats:
            embedding = model_dict[s][1]
            for c in range(embedding.shape[1]):
                score = roc_auc_score(df[t], embedding[:, c])
                r = {
                    'topic': t,
                    'strategy': s,
                    'column': c,
                    'score': score,
                }
                records.append(r)
    return pd.DataFrame(records)


def main(msg, feature):
    infile = Path("../data/full_04_01_2020/kwds.jsonl")
    df = load_data(infile)
    df = df.pipe(get_textrank_kwds).pipe(get_rake_kwds)
    strats = ['textrank', 'rake']
    model_dict = {}
    for s in strats:
        model_dict[s] = train_lda(df[s])
    LOG.info(f'{msg} and feature is {feature}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input txt file')
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--no-feature', dest='feature', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.i, args.feature)
