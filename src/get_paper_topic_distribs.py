import argparse
import logging
from pathlib import Path
from gensim.models.ldamodel import LdaModel
from db import extract_keyword_from_doc, get_spacy_nlp
from tqdm import tqdm
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(
    lda_model,
    dir_of_txts,
    output_embedding_csv,
    #spacy_nlp_name="en_core_web_sm",
    spacy_nlp_name="en_core_sci_lg",
    batch_size=1000,
    n_process=-1,
):
    LOG.info("Loading lda model and texts.")
    model = LdaModel.load(str(lda_model))
    texts = []
    txt_locs = list(dir_of_txts.iterdir())
    txt_stems = [t.stem for t in txt_locs]
    for p in txt_locs:
        with open(p, "r") as f0:
            texts.append(f0.read())

    LOG.info("Extracting keywords from texts using SingleRank")
    nlp = get_spacy_nlp(spacy_nlp_name)
    pipe = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    pbar = tqdm(pipe, total=len(texts))
    corpus = []
    for doc in pbar:
        lemma_text = " ".join([t.lemma_.lower() for t in doc])
        tokens = []
        for _, k in model.id2word.items():
            if k.lower() in lemma_text:
                count = lemma_text.count(k.lower())
                tokens = tokens + ([k] * count)
            else:
                continue
        bow = model.id2word.doc2bow(tokens)
        corpus.append(bow)
    # Just check for occurrence of the dictionary terms in the lemma_text instead of
    # using the whole text extraction

    LOG.info("Running inference to get topic distributions.")
    tc = model.get_document_topics(corpus, minimum_probability=0)
    embedding = np.vstack([[v for t, v in r] for r in tqdm(tc)])
    df = pd.DataFrame(embedding)
    df.index = txt_stems

    LOG.info(f"Writing topic probabilities to {output_embedding_csv}.")
    df.to_csv(output_embedding_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get topic distributions for ")
    parser.add_argument(
        "--lda_model", type=Path, help="LdaModel with which to make predictions"
    )
    parser.add_argument("--dir_of_txts", type=Path, help="Directory of text files")
    parser.add_argument(
        "--output_embedding_csv",
        type=Path,
        help="Topic distribution predictions for provided documents",
    )
    args = parser.parse_args()
    main(args.lda_model, args.dir_of_txts, args.output_embedding_csv)
