import json
import logging
import re
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import torch
from enstop import PLSA, EnsembleTopics
from enstop.utils import coherence
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models import LdaMulticore, LdaModel
from gensim.models.coherencemodel import CoherenceModel
from scipy.io import mmwrite, mmread
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_feature_matrix(lim_kwds_df):
    LOG.info("Making feature matrix...")
    doc_to_kwd = (
        lim_kwds_df.explode("doc_id_list").groupby("doc_id_list").agg({"stem": list})
    )
    mat_id_to_doc_id = (
        doc_to_kwd.reset_index().reset_index().loc[:, ["index", "doc_id_list"]]
    )

    dct = Dictionary(doc_to_kwd["stem"])
    corpus = [dct.doc2bow(text) for text in tqdm(doc_to_kwd["stem"])]
    mat_id_to_doc_id.columns = ["matrix_row_index", "doc_id"]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(doc_to_kwd["stem"])  # inverse for docs x kwds
    return X, mlb, mat_id_to_doc_id, dct, corpus


def __num_dist_rows__(array, ndigits=2):
    return array.shape[0] - int((pd.DataFrame(array).sum(axis=1) < 0.99).sum())


def topic_model_viz(model, mlb, mdoc_lens, viz_loc):
    term_freq = np.array(model.training_data_.sum(axis=0))[0]
    data = {
        "topic_term_dists": model.components_,
        "doc_topic_dists": model.embedding_,
        "vocab": mlb.classes_,
        "term_frequency": term_freq,
        "doc_lengths": np.array(mdoc_lens),
    }
    pyLDAvis._prepare.__num_dist_rows__ = __num_dist_rows__
    LOG.info("Preparing data for pyLDAvis")
    viz_data = pyLDAvis.prepare(**data, sort_topics=False)
    LOG.info(f"Writing visualization to {viz_loc}")
    pyLDAvis.save_html(viz_data, str(viz_loc))
    return viz_loc


def topic_model_viz_gensim(corpus, dct, lda, doc_lens, viz_loc):
    LOG.info("Getting documents topic distributions.")
    tc = lda.get_document_topics(corpus, minimum_probability=0)
    tmp = [[v for t, v in r] for r in tqdm(tc)]
    tmp_a = np.vstack(tmp)

    term_freqs = np.array([dct.cfs[i] for i in range(lda.num_terms)])
    vocab = np.array([lda.id2word[i] for i in range(lda.num_terms)])

    data = {
        "topic_term_dists": lda.get_topics(),
        "doc_topic_dists": tmp_a,
        "vocab": vocab,
        "term_frequency": term_freqs,
        "doc_lengths": doc_lens,
    }
    pyLDAvis._prepare.__num_dist_rows__ = __num_dist_rows__
    LOG.info("Preparing data for pyLDAvis")
    viz_data = pyLDAvis.prepare(**data, sort_topics=False)

    LOG.info(f"Writing visualization to {viz_loc}")
    pyLDAvis.save_html(viz_data, str(viz_loc))


def tmodel_to_tboard(X, model, doc_ids):
    writer = SummaryWriter()
    labels = model.embedding_.argmax(axis=1).tolist()
    meta = list(zip(doc_ids, [str(l) for l in labels]))
    LOG.info("Writing to tensorboard.")
    kwd_embedding = np.array(X.todense())
    writer.add_embedding(
        kwd_embedding,
        tag="keyword_embedding",
        metadata=meta,
        label_img=None,
        metadata_header=["doc_id", "topic"],
    )
    writer.close()
    LOG.info('Use "tensorboard --logdir runs" command to see visualization.')
    return model


def plot_coherence(topic_range, coherences, show=False):
    LOG.info("Plotting coherences.")
    plt.plot(topic_range, coherences)
    plt.xlabel("n_topics")
    plt.ylabel("coherence")
    plt.title("Model Coherence vs Number of Topics")
    if show:
        plt.show()
    return plt.gcf()


def run_topic_models_inner(
    X, mat_doc_id_map, plot_loc, tmodels_dir, tboard=False, alg="plsa"
):
    labels = mat_doc_id_map["doc_id"].tolist()
    # TODO: add train and test? But its clustering so maybe no?
    if alg == "plsa":
        TopicModel = PLSA
    elif alg == "lda":
        TopicModel = LatentDirichletAllocation
    elif alg == "enstop":
        TopicModel = EnsembleTopics
    else:
        ValueError(f'Must choose algorithm from "plsa", "lda", and "enstop".')

    topic_range = [10, 20, 30]
    coherences = []
    # TODO: instead of appending, directly write to dir of tmodels with n_topics

    LOG.info(f"Training topic models and writing to {tmodels_dir}")
    topic_pbar = tqdm(topic_range)
    for n in topic_pbar:
        topic_pbar.set_description(f"n_topics: {n}")
        # model = EnsembleTopics(n_components=n, n_jobs=12).fit(X)
        model = TopicModel(n_components=n).fit(X)
        joblib.dump(model, tmodels_dir / f"topics_{n}.jbl")
        if tboard:  # will slow things down by A LOT, also does not seem to work yet
            tmodel_to_tboard(X, model, labels)
        if alg == "plsa":
            c = model.coherence()
        elif alg == "enstop":
            c = model.coherence()
        elif alg == "lda":
            c = coherence(model.components_, n, X, n_words=20)
        else:
            ValueError(f'Must choose algorithm from "plsa", "lda", and "enstop".')
        coherences.append(c)
    fig = plot_coherence(topic_range, coherences)
    LOG.info(f"Writing plot to {plot_loc}.")
    fig.savefig(str(plot_loc))
    return plot_loc


def get_doc_length(line):
    record = json.loads(line)
    nanlen = lambda x: len(x) if x is not None else 0
    doc_len = nanlen(record["title"]) + nanlen(record["abstract"])
    return doc_len


def get_bib_titles(line):
    record = json.loads(line)
    b = record["bibcode"]
    t = record["title"]
    return (b, t)


def get_bibcodes_from_file(infile):
    with open(infile, "r") as f0:
        bts = list(zip(*[get_bib_titles(line) for line in tqdm(f0)]))
    return bts


def get_doc_len_from_file(infile,):
    with open(infile, "r") as f0:
        doc_lens = [get_doc_length(line) for line in tqdm(f0)]
    return doc_lens


@click.group()
def cli():
    pass


@cli.command()
@click.option("--infile", default=Path)
@click.option("--outfile", default=Path)
def prepare_for_neural_lda(infile, outfile):
    LOG.info(f"Writing titles and abstracts from {infile} to {outfile}.")
    with open(infile, "r") as f_in, open(outfile, "w") as f_out:
        for in_line in tqdm(f_in.read().splitlines()):
            record = json.loads(in_line)
            out_txt = record["title"] + ". " + record["abstract"]
            f_out.write(out_txt.replace("\n", "").strip())
            f_out.write("\n")
    return outfile


@cli.command()
@click.option("--in_docs", type=Path)
@click.option("--lda_model_loc", type=Path)
@click.option("--n_topics", tpye=int, default=10)
@click.option("--num_epochs", tpye=int, default=10)
def run_neural_lda(in_docs, lda_model_loc, n_topics=10,num_epochs=10):
    from contextualized_topic_models.models.ctm import CTM
    from contextualized_topic_models.utils.data_preparation import TextHandler
    from contextualized_topic_models.utils.data_preparation import (
        bert_embeddings_from_file,
    )
    from contextualized_topic_models.datasets.dataset import CTMDataset
    from contextualized_topic_models.evaluation.measures import CoherenceNPMI

    LOG.info("Creating vocabulary.")
    handler = TextHandler(in_docs)
    handler.prepare()  # create vocabulary and training data

    LOG.info("Generating BERT embeddings.")
    training_bert = bert_embeddings_from_file(
        in_docs, "distiluse-base-multilingual-cased"
    )
    training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

    LOG.info("Training model")
    ctm = CTM(
        input_size=len(handler.vocab),
        bert_input_size=512,
        inference_type="combined",
        n_components=n_topics,
        num_epochs=num_epochs,
    )
    ctm.fit(training_dataset)  # run the model

    LOG.info(f"Writing model to {lda_model_loc}")
    torch.save(ctm, lda_model_loc)

    with open(in_docs, "r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()]  # load text for NPMI

    npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
    s = npmi.score()
    LOG.info(f"Coherence: {s}")


@cli.command()
@click.option("--norm_loc", type=Path)
@click.option("--mat_loc", type=Path)
@click.option("--mlb_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--corp_loc", type=Path)
def prepare_features(norm_loc, mat_loc, mlb_loc, map_loc, dct_loc, corp_loc):
    """
    Create document term matrix
    """
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    lim_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)
    X, mlb, mat_doc_id_map, dct, corpus = get_feature_matrix(lim_kwds_df)

    LOG.info("Writing matrix, multilabel binarizer, and matrix to doc id mapping.")

    LOG.info(f"Writing doc feature matrix to  {mat_loc}")
    mmwrite(str(mat_loc), X)

    LOG.info(f"Writing multilabel binarizer to {mlb_loc}")
    joblib.dump(mlb, mlb_loc)

    LOG.info(f"Writing matrix to doc id mapping to {map_loc}")
    mat_doc_id_map.to_csv(map_loc)

    LOG.info(f"Writing gensim dictionary to {dct_loc}")
    dct.save(str(dct_loc))

    LOG.info(f"Writing gensim corpus to {corp_loc}")
    MmCorpus.serialize(str(corp_loc), corpus)


@cli.command()
@click.option("--plot_loc", type=Path)
@click.option("--mat_loc", type=Path)
@click.option("--mlb_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--tmodels_dir", type=Path)
@click.option("--alg", type=str, default="plsa")
def run_topic_models(plot_loc, mat_loc, mlb_loc, map_loc, tmodels_dir, alg="plsa"):
    """
    Create topic models and write to tensorboard
    """
    LOG.info(f"Reading doc feature matrix from  {mat_loc}")
    X = mmread(str(mat_loc)).tocsr()
    LOG.info(f"Read matrix to doc id mapping from {map_loc}")
    mat_doc_id_map = pd.read_csv(map_loc, index_col=0)
    run_topic_models_inner(X, mat_doc_id_map, plot_loc, tmodels_dir, alg=alg)


@cli.command()
@click.option("--plot_loc", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--corp_loc", type=Path)
@click.option("--tmodels_dir", type=Path)
def run_gensim_lda_mult(plot_loc, dct_loc, corp_loc, tmodels_dir):
    dct = Dictionary.load(str(dct_loc))
    corpus = MmCorpus(str(corp_loc))

    topic_range = list(range(10, 2000, 10))
    # topic_range = [20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500]
    coherences = []
    pbar = tqdm(topic_range)
    for n_topics in pbar:
        pbar.set_description(f"n_topics: {n_topics}")
        lda = LdaMulticore(corpus, id2word=dct, num_topics=n_topics)
        cm = CoherenceModel(model=lda, corpus=corpus, coherence="u_mass")
        coherence = cm.get_coherence()  # get coherence value
        coherences.append(coherence)
        # Save model to disk.
        out_model = tmodels_dir / f"gensim_topic_model{n_topics}"
        lda.save(str(out_model))
    fig = plot_coherence(topic_range, coherences)
    LOG.info(f"Writing plot to {plot_loc}.")
    fig.savefig(str(plot_loc))
    return plot_loc


@cli.command()
@click.option("--plot_loc", type=Path)
@click.option("--corp_loc", type=Path)
@click.option("--tmodels_dir", type=Path)
def get_gensim_coherences(plot_loc, corp_loc, tmodels_dir):
    corpus = MmCorpus(str(corp_loc))

    pattern = re.compile("gensim_topic_model*")
    model_locs = set()
    for l in tmodels_dir.iterdir():
        s = l.stem
        if pattern.search(s):
            model_locs.add(tmodels_dir / s.split(".")[0])

    topic_range = []
    coherences = []
    pbar = tqdm(model_locs)
    for in_model in pbar:
        pbar.set_description(f"in_model: {in_model}")
        lda = LdaModel.load(str(in_model))
        topic_range.append(lda.num_topics)
        cm = CoherenceModel(model=lda, corpus=corpus, coherence="u_mass")
        coherence = cm.get_coherence()  # get coherence value
        coherences.append(coherence)
        # Save model to disk.
    fig = plot_coherence(topic_range, coherences)
    LOG.info(f"Writing plot to {plot_loc}.")
    fig.savefig(str(plot_loc))
    return plot_loc


@cli.command()
@click.option("--infile", type=Path)
@click.option("--tmodel_dir", type=Path)
@click.option("--n", type=int, default=7)
@click.option("--mlb_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--tmodel_viz_loc", type=Path)
def visualize_topic_models(infile, tmodel_dir, n, mlb_loc, map_loc, tmodel_viz_loc):
    tmodel_loc = tmodel_dir / f"topics_{n}.jbl"
    LOG.info(f"Counting document lengths from {infile}.")
    doc_lens = get_doc_len_from_file(infile)
    LOG.info(f"Loading topic model from {tmodel_loc}")
    tmodel = joblib.load(tmodel_loc)
    LOG.info(f"Loading multilabel binarizer from {mlb_loc}")
    mlb = joblib.load(mlb_loc)
    LOG.info(f"Reading matrix to doc id mapping from {map_loc}")
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)
    mdoc_lens = [doc_lens[i] for i in mat_id_to_doc_id["matrix_row_index"]]
    topic_model_viz(tmodel, mlb, mdoc_lens, tmodel_viz_loc)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--tmodel_dir", type=Path)
@click.option("--n", type=int, default=7)
@click.option("--in_corpus", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--tmodel_viz_loc", type=Path)
def visualize_gensim_topic_models(
    infile, tmodel_dir, n, in_corpus, dct_loc, map_loc, tmodel_viz_loc
):
    LOG.info("Loading corpus, dictionary, lda model, and document map")
    corpus = MmCorpus(str(in_corpus))
    dct = Dictionary.load(str(dct_loc))
    lda = LdaModel.load(str(tmodel_dir / f"gensim_topic_model{n}"))
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)

    doc_lens = get_doc_len_from_file(infile)
    mdoc_lens = [doc_lens[i] for i in mat_id_to_doc_id["matrix_row_index"]]
    topic_model_viz_gensim(corpus, dct, lda, mdoc_lens, tmodel_viz_loc)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--tmodel_dir", type=Path)
@click.option("--n", type=int)
@click.option("--mlb_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--topic_to_bibcodes_loc", type=Path)
def explore_topic_models(
    infile, tmodel_dir, n, mlb_loc, map_loc, topic_to_bibcodes_loc
):
    tmodel_loc = tmodel_dir / f"topics_{n}.jbl"
    LOG.info(f"Loading topic model from {tmodel_loc}")
    tmodel = joblib.load(tmodel_loc)

    LOG.info(f"Loading multilabel binarizer from {mlb_loc}")
    mlb = joblib.load(mlb_loc)

    LOG.info(f"Reading matrix to doc id mapping from {map_loc}")
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)

    LOG.info(f"Getting document bibcodes from {infile}.")
    bibcodes, titles = get_bibcodes_from_file(infile)
    mdoc_bibs = [bibcodes[i] for i in mat_id_to_doc_id["doc_id"]]
    mdoc_titles = [titles[i] for i in mat_id_to_doc_id["doc_id"]]

    df = pd.DataFrame(tmodel.embedding_)
    df["bibcode"] = mdoc_bibs
    df["titles"] = mdoc_titles
    cols = df.columns[-2:].tolist() + df.columns[0:-2].tolist()
    df = df[cols]

    LOG.info("Reorder by descending topic scores where given topic is max")
    new_df = pd.DataFrame()
    top_topics = df.iloc[:, 2:].values.argmax(axis=1)
    for t in tqdm(range(tmodel.embedding_.shape[1])):
        new_df = new_df.append(
            df.iloc[top_topics == t, :].sort_values(t, ascending=False)
        )

    LOG.info(f"Writing bibcodes to {topic_to_bibcodes_loc}")
    new_df.to_csv(topic_to_bibcodes_loc)


if __name__ == "__main__":
    cli()
