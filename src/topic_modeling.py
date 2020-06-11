import json
import h5py
import dask
from collections import defaultdict
import logging
import os
import re
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import gensim
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import torch
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

# from enstop import PLSA, EnsembleTopics
# from enstop.utils import coherence
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models import LdaMulticore, LdaModel
from gensim.models.coherencemodel import CoherenceModel
from scipy.io import mmwrite, mmread
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter
from tqdm import tqdm

from extract_keywords import binarize_years, get_stem_aggs

# LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if np.iscomplexobj(obj):
            return abs(obj)
        return json.JSONEncoder.default(self, obj)


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


def topic_model_viz_gensim(embedding, dct, lda, doc_lens, viz_loc):

    term_freqs = np.array([dct.cfs[i] for i in range(lda.num_terms)])
    vocab = np.array([lda.id2word[i] for i in range(lda.num_terms)])

    data = {
        "topic_term_dists": lda.get_topics(),
        "doc_topic_dists": embedding,
        "vocab": vocab,
        "term_frequency": term_freqs,
        "doc_lengths": doc_lens,
    }
    pyLDAvis.utils.NumPyEncoder = NumPyEncoder
    pyLDAvis._prepare.__num_dist_rows__ = __num_dist_rows__
    LOG.info("Preparing data for pyLDAvis")
    # viz_data = pyLDAvis.prepare(**data, sort_topics=False)
    viz_data = pyLDAvis.gensim.prepare(**data, sort_topics=False, mds="mmds")

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


def plot_coherence(df, c_measures, show=False):
    LOG.info("Plotting coherences.")
    for c in c_measures:
        plt.plot(df["n_topics"], df[c], label=c)
    plt.xlabel("n_topics")
    plt.ylabel("coherence")
    plt.title("Model Coherence vs Number of Topics")
    plt.legend()
    if show:
        plt.show()
    return plt.gcf()


def run_topic_models_inner(
    X, mat_doc_id_map, plot_loc, tmodels_dir, tboard=False, alg="plsa"
):
    labels = mat_doc_id_map["doc_id"].tolist()
    # TODO: add train and test? But its clustering so maybe no?
    if alg == "plsa":
        # TopicModel = PLSA
        ValueError(f"Enstop deprecated")
    elif alg == "lda":
        TopicModel = LatentDirichletAllocation
    elif alg == "enstop":
        # TopicModel = EnsembleTopics
        ValueError(f"Enstop deprecated")
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
            # TODO: add new coherence metric here.
            ValueError(f"Enstop deprecated")
            # c = coherence(model.components_, n, X, n_words=20)
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
    return b, t


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


def get_bow_term_doc_matrix(dct, corpus):
    bow = np.zeros((len(corpus), len(dct)), dtype=int)
    for i, bow_inds in enumerate(tqdm(corpus)):
        inds, vals = list(zip(*bow_inds))
        bow[i, inds] = np.array(vals)
    return bow


def load_ctm(model_file):
    """
    Load a previously trained model.

    Args
        model_dir: directory where models are saved.
        epoch: epoch of model to load.
    """
    with open(model_file, "rb") as model_dict:
        checkpoint = torch.load(model_dict)
    ctm = CTM(
        input_size=checkpoint["dcue_dict"]["input_size"],
        bert_input_size=checkpoint["dcue_dict"]["bert_size"],
        batch_size=checkpoint["dcue_dict"]["batch_size"],
        inference_type="combined",
    )
    for (k, v) in checkpoint["dcue_dict"].items():
        setattr(ctm, k, v)

    # self._init_nn() #TODO: figure out what this does and implement?
    ctm.model.load_state_dict(checkpoint["state_dict"])
    return ctm


def ctm_save(ctm, models_dir=None):
    """
    Save model.

    Args
        models_dir: path to directory for saving NN models.
    """
    if (ctm.model is not None) and (models_dir is not None):

        model_dir = ctm._format_file()
        if not os.path.isdir(os.path.join(models_dir, model_dir)):
            os.makedirs(os.path.join(models_dir, model_dir))

        filename = "epoch_{}".format(ctm.nn_epoch) + ".pth"
        fileloc = os.path.join(models_dir, model_dir, filename)
        torch.save(
            {"state_dict": ctm.model.state_dict(), "dcue_dict": ctm.__dict__},
            fileloc,
            pickle_protocol=4,
        )


@cli.command()
@click.option("--in_docs", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--corp_loc", type=Path)
@click.option("--lda_model_dir", type=Path)
@click.option("--n_topics", type=int, default=10)
@click.option("--num_epochs", type=int, default=10)
def run_neural_lda(
    in_docs, dct_loc, corp_loc, lda_model_dir, n_topics=10, num_epochs=10
):

    LOG.info("Loading dictionary and corpus.")
    dct = Dictionary.load(str(dct_loc))
    corpus = MmCorpus(str(corp_loc))
    idx2token = {i: dct[i] for i in range(len(dct))}
    LOG.info("Making bow term doc matrix")
    bow = get_bow_term_doc_matrix(dct, corpus)

    LOG.info("Generating BERT embeddings.")
    training_bert = bert_embeddings_from_file(
        in_docs, "distiluse-base-multilingual-cased"
    )
    training_dataset = CTMDataset(bow, training_bert, idx2token)

    LOG.info("Training model")
    ctm = CTM(
        input_size=len(dct),
        bert_input_size=512,
        batch_size=32,
        inference_type="combined",
        n_components=n_topics,
        num_epochs=num_epochs,
        num_data_loader_workers=1,
    )
    ctm.fit(training_dataset)  # run the model
    ctm.model.cpu()
    LOG.info(f"Writing model to {lda_model_dir}")
    ctm_save(ctm, lda_model_dir)

    ctm.model.cuda()
    embedding = ctm.predict(training_dataset)
    embedding_loc = lda_model_dir / "embedding.pt"
    LOG.info(f"Writing embedding to {embedding_loc}")
    torch.save(embedding, embedding_loc)

    coh = CoherenceModel(
        topics=ctm.get_topic_lists(10),
        corpus=corpus,
        dictionary=dct,
        coherence="u_mass",
        topn=10,
    )
    s = coh.get_coherence()
    LOG.info(f"Coherence: {s}")


@cli.command()
@click.option("--docs_loc", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--corp_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--token_loc", type=Path)
@click.option("--no_below", type=int)
@click.option("--no_above", type=float)
def prepare_gensim_features(
    docs_loc, dct_loc, corp_loc, map_loc, token_loc, no_below=300, no_above=0.25
):
    # TODO: save the corpus and dictionary
    df = pd.read_json(docs_loc, orient="records", lines=True)
    df = df.dropna(subset=["bibcode", "title", "abstract"])
    texts = df["title"] + ". " + df["abstract"]
    tokens = texts.apply(
        lambda x: [v for v in gensim.summarization.textcleaner.tokenize_by_word(x)]
    )

    dct = gensim.corpora.Dictionary(tokens)
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dct.doc2bow(t) for t in tokens]
    mat_id_to_bibcode = df.reset_index().reset_index().iloc[:, :2]
    mat_id_to_bibcode.columns = ["matrix_row_index", "doc_id"]

    LOG.info(f"Writing dictionary to {dct_loc}")
    dct.save(str(dct_loc))

    LOG.info(f"Writing corpus to {corp_loc}")
    MmCorpus.serialize(str(corp_loc), corpus)

    LOG.info(f"Writing matrix id to df id to {map_loc}")
    mat_id_to_bibcode.to_csv(map_loc)

    LOG.info(f"Writing tokens to {token_loc}")
    with open(token_loc, "w") as f0:
        for token_set in tokens:
            f0.write(json.dumps(token_set))
            f0.write("\n")
    mat_id_to_bibcode.to_csv(map_loc)

    return dct_loc, corp_loc, map_loc


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


def make_tmodel_n(pbar, corpus, dct, n_topics, c_measures, texts, tmodels_dir):
    pbar.set_description(f"n_topics: {n_topics}")
    lda = LdaModel(
        corpus,
        id2word=dct,
        num_topics=n_topics,
        passes=5,
        iterations=200,
        chunksize=100_000,
        # passes=1,
        # iterations=50,
        # eval_every=1,
        eval_every=10,
        alpha="auto",
        eta="auto",
    )
    out_model = tmodels_dir / f"gensim_topic_model{n_topics}"
    lda.save(str(out_model))
    pbar.update(1)
    return lda


def get_coherence_df(ldas, topic_range, c_measures, texts=None, corpus=None, dct=None):
    coherences = defaultdict(list)
    coh_pbar = tqdm(zip(ldas, topic_range), total=len(ldas))
    for lda, n in coh_pbar:
        for c in c_measures:
            coh_pbar.set_description(f"n_topics={n} | measure={c}")
            cm = CoherenceModel(
                model=lda, texts=texts, corpus=corpus, coherence=c, dictionary=dct
            )
            coherence = cm.get_coherence()  # get coherence value
            coherences[c].append(coherence)

    df = pd.DataFrame(
        {
            **{
                "model": f"{lda.__class__.__module__}.{lda.__class__.__name__}",
                "n_topics": topic_range,
            },
            **coherences,
        }
    )
    return df


# TODO: this code is getting messy. Looking like there should be a class
def run_gensim_lda_mult_inner(
    topic_range, dct, tmodels_dir, c_measures, corpus=None, texts=None
):
    jobs = []
    pbar = tqdm(topic_range)
    for n_topics in topic_range:
        j = dask.delayed(make_tmodel_n)(
            pbar, corpus, dct, n_topics, c_measures, texts, tmodels_dir
        )
        jobs.append(j)
    ldas = dask.compute(jobs)[0]
    df = get_coherence_df(ldas, topic_range, c_measures, texts=texts, corpus=corpus)
    return df


@cli.command()
@click.option("--plot_loc", type=Path)
@click.option("--tokens_loc", type=Path, default="")
@click.option("--topic_range_loc", type=Path)
@click.option("--tmodels_dir", type=Path)
@click.option("--coherence_loc", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--corp_loc", type=Path)
def run_gensim_lda_mult(
    plot_loc, tokens_loc, topic_range_loc, tmodels_dir, coherence_loc, dct_loc, corp_loc
):
    LOG.info(f"Loading dictionary from {dct_loc}")
    dct = Dictionary.load(str(dct_loc))
    LOG.info(f"Loading corpus from {corp_loc}")
    corpus = MmCorpus(str(corp_loc))
    if tokens_loc != Path("."):
        with open(tokens_loc, "r") as f0:
            # Load only the keywords without their scores from each line.
            tokens = [
                [k for k, v in json.loads(line)] for line in f0.read().splitlines()
            ]
    else:
        tokens = None
    with open(topic_range_loc, "r") as f0:
        topic_range = json.load(f0)
    if len(topic_range) == 0:
        ValueError("Topic range is an empty list.")

    c_measures = ["u_mass", "c_v"]
    df = run_gensim_lda_mult_inner(
        topic_range,
        dct,
        tmodels_dir,
        c_measures=c_measures,
        corpus=corpus,
        texts=tokens,
    )
    fig = plot_coherence(df, c_measures)

    LOG.info(f"Writing coherences to {coherence_loc}.")
    df.to_csv(coherence_loc)

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


def write_topic_distributions(df, loc):
    """
    Write topic distributions to h5py

    Args:
        df: dataframe with first column of bibcodes and topic distributions for all after
        loc: where to save the h5py file

    Returns:
        loc: the location where the file has been saved
    """
    vals = df.iloc[:, 1:]
    tmaxes = vals.values.argmax(axis=1)
    dt = h5py.string_dtype()

    with h5py.File(loc, "w") as f0:
        f0.create_dataset("bibcodes", (df.shape[0],), dtype=dt, data=df["bibcode"])
        f0.create_dataset(
            "topic_distribution",
            (vals.shape[0], vals.shape[1]),
            dtype=np.float64,
            data=vals,
        )
        f0.create_dataset("topic_maxes", (tmaxes.shape[0],), dtype=np.int, data=tmaxes)
        f0.create_dataset(
            "dist_to_doc_index", (df.shape[0],), dtype=np.int, data=df.index.values
        )

    return loc


@cli.command()
@click.option("--infile", type=Path)
@click.option("--tmodel_dir", type=Path)
@click.option("--n", type=int, default=7)
@click.option("--in_corpus", type=Path)
@click.option("--dct_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--tmodel_viz_loc", type=Path)
@click.option("--viz_data_loc", type=Path)
@click.option("--topic_to_bibcodes_loc", type=Path)
@click.option("--topic_cohs_loc", type=Path)
def visualize_gensim_topic_models(
    infile,
    tmodel_dir,
    n,
    in_corpus,
    dct_loc,
    map_loc,
    tmodel_viz_loc,
    viz_data_loc,
    topic_to_bibcodes_loc,
    topic_cohs_loc,
):
    LOG.info("Loading corpus, dictionary, lda model, and document map")
    corpus = MmCorpus(str(in_corpus))
    dct = Dictionary.load(str(dct_loc))
    lda = LdaModel.load(str(tmodel_dir / f"gensim_topic_model{n}"))
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)

    cm = CoherenceModel(model=lda, corpus=corpus, coherence="u_mass")
    coh_per_topic = cm.get_coherence_per_topic()

    # Just counting number of terms which appear, not the frequency of them
    # Do this because we are only counting occurrence for LDA, not using frequency
    # This could be a problem. Might need to change that.
    # Counting freq would be like:
    # mdoc_lens = [sum([v[1] for v in corpus[i]]) for i in mat_id_to_doc_id["matrix_row_index"]]
    # mdoc_lens = [len(corpus[i]) for i in mat_id_to_doc_id["matrix_row_index"]]

    LOG.info("Getting documents topic distributions.")
    tc = lda.get_document_topics(corpus, minimum_probability=0)
    embedding = np.vstack([[v for t, v in r] for r in tqdm(tc)])
    new_df = get_bibcodes_with_embedding(infile, embedding, mat_id_to_doc_id)

    LOG.info(f"Writing topic coherences to {topic_cohs_loc}")
    pd.DataFrame(coh_per_topic).to_csv(topic_cohs_loc)

    LOG.info(f"Writing bibcodes to {topic_to_bibcodes_loc}")
    write_topic_distributions(new_df, topic_to_bibcodes_loc)

    viz_data = pyLDAvis.gensim.prepare(lda, corpus, dct, sort_topics=False, mds="mmds", start_index=0)
    LOG.info(f"Writing visualization data to {viz_data_loc}")
    with open(viz_data_loc, 'w') as f0:
        json.dump(viz_data.to_json(), f0)
    LOG.info(f"Writing visualization to {tmodel_viz_loc}")
    pyLDAvis.save_html(viz_data, str(tmodel_viz_loc), ldavis_url="/static/js/ldavis.js") # TODO: fix this path


def get_bibcodes_with_embedding(infile, embedding, mat_id_to_doc_id):
    LOG.info(f"Getting document bibcodes from {infile}.")
    bibcodes, titles = get_bibcodes_from_file(infile)
    mdoc_bibs = [bibcodes[i] for i in mat_id_to_doc_id["doc_id"]]

    df = pd.DataFrame(embedding)
    df.insert(0, "bibcode", mdoc_bibs)

    LOG.info("Reorder by descending topic scores where given topic is max")
    new_df = pd.DataFrame()
    top_topics = df.iloc[:, 2:].values.argmax(axis=1)
    for t in tqdm(range(embedding.shape[1])):
        new_df = new_df.append(
            df.iloc[top_topics == t, :].sort_values(t, ascending=False)
        )
    return new_df


@cli.command()
@click.option("--records_loc", type=Path)
@click.option("--in_bib", type=Path)
@click.option("--topic_cohs_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--out_years", type=Path)
@click.option("--year_min", type=int, default=0)
def get_topic_years(records_loc, in_bib, topic_cohs_loc, map_loc, out_years, year_min):
    LOG.info("Reading data...")
    df = pd.read_json(records_loc, orient="records", lines=True)
    with h5py.File(in_bib, "r") as f0:
        bibs = f0["bibcodes"][:]
        vals = f0["topic_distribution"][:]
        dind = f0["dist_to_doc_index"][:]
    bib_df = pd.DataFrame(vals)
    bib_df.insert(0, "bibcode", bibs)
    bib_df.index = dind
    map_df = pd.read_csv(map_loc, index_col=0)
    coh_df = pd.read_csv(topic_cohs_loc, index_col=0)

    LOG.info("Transforming...")
    topics = bib_df.set_index("bibcode").values.argmax(axis=1)  # Could add thresh
    # Thresh in addition to argmax? Too limiting but everything is complicate if one doc
    # can be a part of multiple topics for the counts over time.
    ndf = pd.DataFrame(topics)
    ndf.index = bib_df.index
    ndf.columns = [
        "stem"
    ]  # Lost information about matrix row index with the bibcode version
    # need to get the matrix location from the bibcode now
    ndf = ndf.join(map_df.set_index("matrix_row_index"))
    ndf = ndf.sort_values("doc_id")
    ndf["year"] = df.loc[ndf["doc_id"], "year"].values
    ndf = ndf.loc[ndf['year'] >= year_min]
    # why wrong if I don't use values? Because automatically setting index?
    ndf["nasa_afil"] = df.loc[ndf["doc_id"], "nasa_afil"].values
    ndf = ndf.dropna().copy()
    ndf["year"] = ndf["year"].astype(int)
    ndf["score"] = coh_df.loc[ndf["stem"].tolist()].iloc[:, 0].tolist()
    ndf["keyword"] = ["" for _ in range(ndf.shape[0])]
    # Here have multiple lines for each doc and it might work.
    bndf = binarize_years(ndf)
    agg_df = get_stem_aggs(bndf).reset_index()
    agg_df["stem"] = agg_df["stem"].astype(str)

    LOG.info(f"Writing to {out_years}")
    agg_df.to_json(out_years, orient="records", lines=True)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--tmodel_dir", type=Path)
@click.option("--n", type=int)
@click.option("--mlb_loc", type=Path)
@click.option("--map_loc", type=Path)
@click.option("--topic_to_bibcodes_loc", type=Path)
@click.option("--topic_to_years_loc", type=Path)
def explore_topic_models(
    infile, tmodel_dir, n, mlb_loc, map_loc, topic_to_bibcodes_loc, topic_to_years_loc,
):
    tmodel_loc = tmodel_dir / f"topics_{n}.jbl"
    LOG.info(f"Loading topic model from {tmodel_loc}")
    tmodel = joblib.load(tmodel_loc)

    LOG.info(f"Loading multilabel binarizer from {mlb_loc}")
    mlb = joblib.load(mlb_loc)

    LOG.info(f"Reading matrix to doc id mapping from {map_loc}")
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)

    bib_df = get_bibcodes_with_embedding(infile, tmodel.embedding_, mat_id_to_doc_id)
    LOG.info(f"Writing bibcodes to {topic_to_bibcodes_loc}")
    bib_df.to_csv(topic_to_bibcodes_loc)


if __name__ == "__main__":
    cli()
