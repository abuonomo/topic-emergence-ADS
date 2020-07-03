import json
from collections import defaultdict
from pprint import pformat

import click
import dask
from typing import List
import h5py
import logging
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import yaml
from gensim.corpora import MmCorpus, Dictionary
from gensim.models import LdaModel, CoherenceModel
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from tsfresh import extract_features
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import Manifold
from dtw_time_analysis import dtw_kwds

from db import Paper, Keyword, PaperKeywords

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class TopicModeler:
    def __init__(self, dictionary, corpus):
        self.dictionary = dictionary
        self.corpus = corpus

    def make_all_topic_models(self, topic_range, **kwargs):
        gensim_logger = logging.getLogger("gensim")
        gensim_logger.setLevel(logging.DEBUG)
        LOG.setLevel(logging.DEBUG)

        @dask.delayed
        def make_topic_model(n_topics, c, d, **kwargs):
            LOG.warning(f"Making topic model with {n_topics} topics.")
            lda = LdaModel(c, id2word=d, num_topics=n_topics, **kwargs)
            return lda

        jobs = []
        for t in topic_range:
            j = make_topic_model(t, self.corpus, self.dictionary, **kwargs)
            jobs.append(j)
        ldas = dask.compute(jobs)[0]
        return ldas

    def get_coherence_model(self, model):
        cm = CoherenceModel(
            model, corpus=self.corpus, coherence="u_mass", dictionary=self.dictionary
        )  # only u_mass because tokens out of order and overlapping
        return cm

    def get_coherences(self, lda_models):
        coherences = []
        coh_pbar = tqdm(lda_models)
        for lda_model in coh_pbar:
            coh_pbar.set_description(f"n_topics={lda_model.num_topics}")
            cm = self.get_coherence_model(lda_model)
            coherence = cm.get_coherence()  # get coherence value
            coherences.append(coherence)

        return coherences

    def get_inference(self, model):
        tc = model.get_document_topics(self.corpus, minimum_probability=0)
        embedding = np.vstack([[v for t, v in r] for r in tqdm(tc)])
        return embedding


def read_from_prepared_data(prepared_data_dir):
    in_corp = prepared_data_dir / "corpus.mm"
    in_dct = prepared_data_dir / "dct.mm"
    in_corp2paper = prepared_data_dir / "corp2paper.json"
    in_dct2kwd = prepared_data_dir / "dct2kwd.json"

    corpus = MmCorpus(str(in_corp))
    dictionary = Dictionary.load(str(in_dct))
    with open(in_corp2paper, "r") as f0:
        corp2paper = json.load(f0)
    with open(in_dct2kwd, "r") as f0:
        dct2kwd = json.load(f0)

    return corpus, dictionary, corp2paper, dct2kwd


def cagr(x_row):
    x = x_row.values
    nz_inds = np.nonzero(x)[0]
    if len(nz_inds) == 0:  # If all are 0, set CAGR to 0
        return 0
    else:
        first_nonzero_index = nz_inds[0]
        x = x[first_nonzero_index:]  # Not valid if starts with 0. Becomes inf
        x = x[
            ~np.isnan(x)
        ]  # For normalized time series, NaNs before any occurrence of kwd
    if len(x) < 2:  # If no periods, set CAGR to 0
        return 0
    else:
        ys = x_row.index
        period = max(ys) - min(ys)
        return (x[-1] / x[0]) ** (1 / period) - 1


class VizPrepper:
    def __init__(self):
        self.pyLDAvis_data = None
        self.embedding_df = None
        self.paper_df = None
        self.paper_ids = None
        self.topic_coherences = None
        self.characteristics = None
        self.kmeans = None
        self.manifold = None
        self.kwd_ts_df = None

    def read_hdf(self, viz_data_loc):

        with h5py.File(viz_data_loc, "r") as f0:
            embedding = f0["embedding"][:]
            topic_coherences = f0["topic_coherences"][:]
            paper_ids = f0["paper_ids"][:]
            bibcodes = f0["bibcodes"][:]
            nasa_affiliation = f0["nasa_affiliation"][:]
            years = f0["years"][:]
            keyword_ids = f0["keyword_ids"][:]
            kwd_ts_values = f0["keyword_ts_values"][:]

        self.paper_df = pd.DataFrame(
            {
                "paper_id": paper_ids,
                "bibcode": bibcodes,
                "year": years,
                "nasa_affiliation": nasa_affiliation,
            }
        )
        embedding_df = pd.DataFrame(embedding)
        embedding_df.index = paper_ids
        self.embedding_df = embedding_df

        kwd_ts_df = pd.DataFrame(kwd_ts_values)
        kwd_ts_df.index = keyword_ids
        self.kwd_ts_df = kwd_ts_df

        self.topic_coherences = topic_coherences

    def read_pyldavis_data(self, pyLDAvis_data_loc):
        with open(pyLDAvis_data_loc, "r") as f0:
            self.pyLDAvis_data = json.load(f0)

    def get_time_characteristics(self, min_topic_prob_thresh, year_min, year_max):
        all_time_series = []
        tmp_paper_df = self.paper_df.copy()
        tmp_paper_df["topic"] = -1
        for topic in tqdm(self.embedding_df.columns):
            tf = self.embedding_df.loc[:, topic] >= min_topic_prob_thresh
            # function to include options with argmax as well?
            # tmp_paper_df.loc[tf, 'topic'] = topic
            tmp_paper_df.loc[tmp_paper_df.index[tf], "topic"] = topic
            ids_in_topic = self.embedding_df.index[tf].tolist()
            year_counts = (
                self.paper_df.loc[
                    (self.paper_df["paper_id"].isin(ids_in_topic))
                    & (self.paper_df["year"] >= year_min)
                    & (self.paper_df["year"] <= year_max)
                ]
                .groupby("year")
                .count()["paper_id"]
            )
            year_counts_df = year_counts.reset_index()
            year_counts = year_counts_df.to_records()
            ycd = defaultdict(int, {y: c for _, y, c in year_counts})
            topic_time_series = [
                {"topic": topic, "year": y, "count": ycd[y]}
                for y in range(year_min, year_max)
            ]
            all_time_series = all_time_series + topic_time_series

        if tmp_paper_df.topic.min() == -1:
            raise ValueError("All papers must have an associated topic.")
        ratio_nasa_affiliation = tmp_paper_df.groupby("topic").agg(
            {"nasa_affiliation": "mean"}
        )
        ts_df_long = pd.DataFrame(all_time_series)
        ts_df = ts_df_long.pivot(index="topic", columns="year", values="count")

        features_df = extract_features(
            ts_df_long, column_id="topic", column_sort="year"
        )
        cols = [c.split('count__')[1] for c in features_df.columns]
        features_df.columns = cols
        features_df["coherence_score"] = self.topic_coherences
        features_df["CAGR"] = ts_df.apply(cagr, axis=1)
        features_df['nasa_affiliation'] = ratio_nasa_affiliation

        return ts_df, features_df

    @staticmethod
    def get_dynamic_time_warp_clusters(ts_df):
        dtw_df = dtw_kwds(ts_df)
        visualizer = yellow_plot_kmd(dtw_df)
        n_clusters = visualizer.elbow_value_
        kmeans = dtw_to_tboard(ts_df, dtw_df, c=n_clusters)
        dtw_man = dtw_to_manifold(dtw_df)
        return kmeans, dtw_man


def dtw_to_manifold(dtw_df, out_plot=None):
    LOG.info("Computing tsne manifold.")
    viz = Manifold(manifold="tsne")
    dtw_man = viz.fit_transform(dtw_df)  # Fit the data to the visualizer
    if out_plot is not None:
        LOG.info(f"Writing tsne manifold plot to {out_plot}.")
        viz.show(out_plot)  # Finalize and render the figure
    return dtw_man


def dtw_to_tboard(normed_kwd_years, dtw_df, c=6, lim=1000):
    """Use pre-computed dynamic time warp values and calculate kmeans."""
    # TODO: Used elbow to determine, but not being placed programmatically
    LOG.info(f"Performing kmeans with {c} clusters.")
    m = KMeans(n_clusters=c)
    m.fit(dtw_df.values)
    return m


def yellow_plot_kmd(X, out_plot=None, c_min=2, c_max=20):
    LOG.info(f"Trying kmeans n_clusters from {c_min} to {c_max}")
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(c_min, c_max))
    visualizer.fit(X)  # Fit the data to the visualizer
    LOG.info(f"Writing elbow to {out_plot}.")
    if out_plot is None:
        return visualizer
    else:
        visualizer.show(out_plot)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--prepared_data_dir", type=Path)
@click.option("--config_loc", type=Path)
@click.option("--out_models_dir", type=Path)
@click.option("--out_coh_csv", type=Path)
def make_topic_models(prepared_data_dir, config_loc, out_models_dir, out_coh_csv):
    corpus, dictionary, _, _ = read_from_prepared_data(prepared_data_dir)
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)

    LOG.info(f"Running with config: \n {pformat(config)}")
    tm = TopicModeler(dictionary, corpus)
    models = tm.make_all_topic_models(config["topic_range"], **config["lda"])
    coherences = tm.get_coherences(models)
    n_topics = [m.num_topics for m in models]
    df = pd.DataFrame(zip(n_topics, coherences))

    LOG.info(f"Writing coherence scores to {out_coh_csv}.")
    df.columns = ["num_topics", "coherence (u_mass)"]
    df.to_csv(out_coh_csv)

    out_models_dir.mkdir(exist_ok=True)
    for m in models:
        mp = out_models_dir / f"topic_model{m.num_topics}"
        LOG.info(f"Writing model to {mp}")
        m.save(str(mp))


def get_pby(session, paper_ids: List, batch_size=990):
    batches = range(0, len(paper_ids), batch_size)
    pbar = tqdm(batches)
    all_records = []
    for i in pbar:
        paper_id_batch = paper_ids[i : i + batch_size]
        q = session.query(
            Paper.id, Paper.bibcode, Paper.year, Paper.nasa_affiliation
        ).filter(Paper.id.in_(paper_id_batch))
        all_records = all_records + q.all()
    df = pd.DataFrame(all_records)
    df.columns = ["id", "bibcode", "year", "nasa_affiliation"]
    return df


def get_kwd_ts_df(session, keyword_ids: List, batch_size=990):
    batches = range(0, len(keyword_ids), batch_size)
    pbar = tqdm(batches)
    all_records = []
    for i in pbar:
        kwd_id_batch = keyword_ids[i : i + batch_size]
        q = (
            session.query(PaperKeywords)
            .join(Keyword)
            .filter(Keyword.id.in_(kwd_id_batch))
        )
        records = [{"keyword": pk.keyword.keyword, "year": pk.paper.year} for pk in q]
        all_records = all_records + records
    df = pd.DataFrame(all_records)
    df.columns = ["keyword", "year"]
    df["count"] = 1
    cdf = df.groupby(["keyword", "year"]).count().reset_index()
    kwd_ts_df = cdf.pivot(index="keyword", columns="year", values="count").fillna(0)
    return kwd_ts_df


@cli.command()
@click.option("--db_loc", type=Path)
@click.option("--prepared_data_dir", type=Path)
@click.option("--tmodel_loc", type=Path)
@click.option("--viz_data_dir", type=Path)
def prepare_for_topic_model_viz(db_loc, prepared_data_dir, tmodel_loc, viz_data_dir):
    LOG.info(f"Loading from {prepared_data_dir} and {tmodel_loc}")
    corpus, dictionary, corp2paper, dct2kwd = read_from_prepared_data(prepared_data_dir)

    lda_model = LdaModel.load(str(tmodel_loc))
    tm = TopicModeler(dictionary, corpus)

    LOG.info("Getting embedding and transforming data.")
    embedding = tm.get_inference(lda_model)
    coh_per_topic = tm.get_coherence_model(lda_model).get_coherence_per_topic()
    topic_maxes = embedding.argmax(axis=1)

    viz_data = pyLDAvis.gensim.prepare(
        lda_model,
        tm.corpus,
        tm.dictionary,
        doc_topic_dist=np.matrix(embedding),
        sort_topics=False,
        mds="mmds",
        start_index=0,
    )
    corpus_inds, paper_ids = zip(*corp2paper)
    dct_inds, keyword_ids = zip(*dct2kwd)

    engine = create_engine(f"sqlite:///{db_loc}")
    Session = sessionmaker(bind=engine)
    session = Session()

    LOG.info(f"Reading paper info from database at {db_loc}")
    pby = get_pby(session, paper_ids)
    kwd_ts_df = get_kwd_ts_df(session, keyword_ids)

    LOG.info(f"Writing data to {viz_data_dir}")
    viz_data_dir.mkdir(exist_ok=True)

    viz_data_loc = viz_data_dir / "viz_data.hdf5"
    pyldavis_data_loc = viz_data_dir / "pyLDAvis_data.json"

    dt = h5py.string_dtype()
    with h5py.File(viz_data_loc, "w") as f0:
        f0.create_dataset("embedding", dtype=np.float, data=embedding)
        f0.create_dataset("topic_coherences", dtype=np.float, data=coh_per_topic)
        f0.create_dataset("topic_maxes", dtype=np.int, data=topic_maxes)
        f0.create_dataset("paper_ids", dtype=np.int, data=paper_ids)
        f0.create_dataset(
            "nasa_affiliation", dtype=bool, data=pby["nasa_affiliation"].values
        )
        f0.create_dataset("years", dtype=np.int, data=pby["year"].values)
        f0.create_dataset("bibcodes", dtype=dt, data=pby["bibcode"].values)
        f0.create_dataset("keyword_ids", dtype=np.int, data=keyword_ids)
        f0.create_dataset("keyword_ts_values", dtype=np.int, data=kwd_ts_df.values)
        f0.create_dataset("keywords", dtype=dt, data=kwd_ts_df.index)
    # pby.to_hdf(viz_data_loc, key="paper2bibcode2year")
    LOG.info(viz_data_loc)

    with open(pyldavis_data_loc, "w") as f0:
        json.dump(viz_data.to_json(), f0)
    LOG.info(pyldavis_data_loc)


@cli.command()
@click.option("--viz_data_dir", type=Path)
@click.option("--config_loc", type=Path)
def get_time_chars(viz_data_dir, config_loc):
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)
    viz_data_loc = viz_data_dir / "viz_data.hdf5"

    vp = VizPrepper()
    vp.read_hdf(viz_data_loc)

    ts_df, chars_df = vp.get_time_characteristics(**config["app"])
    kmeans, dtw_man = vp.get_dynamic_time_warp_clusters(ts_df)
    chars_df["kmeans_cluster"] = kmeans.labels_
    chars_df["manifold_x"] = dtw_man[:, 0]
    chars_df["manifold_y"] = dtw_man[:, 1]
    chars_df["count"] = ts_df.sum(axis=1)
    chars_df["score_mean"] = vp.topic_coherences
    chars_df = chars_df.reset_index().rename(columns={"id": "stem"})

    out_chars_loc = viz_data_dir / "time_series_characteristics.csv"
    chars_df.to_csv(out_chars_loc)

    out_ts_loc = viz_data_dir / "topic_years.csv"
    ts_df.to_csv(out_ts_loc)


def get_coherence_plot(df):
    # df = pd.read_csv(infile, index_col=0)
    xlab = "num_topics"
    ylab = "coherence (u_mass)"

    plt.plot(df[xlab], df[ylab], "-o")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title("Topic Model Coherence versus Number of Topics")
    fig = plt.gcf()
    return fig


if __name__ == "__main__":
    cli()
