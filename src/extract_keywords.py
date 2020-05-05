import logging
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List

import RAKE
import click
import numpy as np
import pandas as pd
import spacy
from nltk import PorterStemmer
from sklearn.preprocessing import LabelBinarizer
from spacy.lang.en import STOP_WORDS
from textacy.ke import textrank
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return "".join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def get_keywords_from_text(text: pd.Series) -> List:
    LOG.info(f"Extracting keywords from {text.shape[0]} documents.")
    tqdm.pandas()
    rake = RAKE.Rake(list(STOP_WORDS))

    def f(x):
        if type(x) == str:
            val = rake.run(x, minFrequency=1, minCharacters=3)
        else:
            val = np.nan
        return val

    rake_kwds = text.parallel_apply(f)
    return rake_kwds


def get_singlerank_kwds(text: pd.Series, batch_size=1000, n_process=1) -> List:
    LOG.info(f"Extracting keywords from {text.shape[0]} documents.")
    kwd_lists = []
    pbar = tqdm(
        NLP.pipe(text.replace(np.nan, ""), batch_size=batch_size, n_process=n_process),
        total=len(text),
    )
    for doc in pbar:
        # SingleRank parameters
        kwds = textrank(
            doc,
            normalize='lemma',
            topn=999,
            window_size=10,
            edge_weighting="count",
            position_bias=False,
        )
        kwd_lists.append(kwds)
        pbar.update(1)
    return kwd_lists


def flatten_to_keywords(df, min_thresh=5):
    df = df.pipe(get_kwd_occurences, min_thresh)
    df = df.pipe(stem_kwds)
    df = df.pipe(stem_reduce, min_thresh)
    df = df.pipe(binarize_years)
    kwd_df = df.pipe(get_stem_aggs)
    year_counts = df["year"].value_counts().reset_index()
    year_counts.columns = ["year", "count"]
    kwd_df = kwd_df.reset_index()
    return kwd_df, year_counts


def get_kwd_occurences(df, min_thresh=5, max_thresh=0.7):
    LOG.info("Flattening along document rake_kwds.")
    df = df.copy()
    df = df.applymap(lambda x: np.nan if x is None else x)
    kwd_inds = df["rake_kwds"].apply(lambda x: type(x) is list)
    rake_kwds = df["rake_kwds"][kwd_inds]
    all_kwds = [(i, k[0], k[1]) for i, r in zip(rake_kwds.index, rake_kwds) for k in r]

    kwd_df = pd.DataFrame(all_kwds)
    kwd_df.columns = ["doc_id", "keyword", "rake_score"]
    kwd_df["year"] = df.loc[kwd_df["doc_id"], "year"].tolist()
    kwd_df["nasa_afil"] = df.loc[kwd_df["doc_id"], "nasa_afil"].tolist()
    ta = df["title"] + "||" + df["abstract"]  # pass this from join func
    ta = ta[ta.apply(lambda x: type(x) == str)]
    n_docs = len(kwd_df.doc_id.unique())
    max_thresh_int = np.ceil(max_thresh * n_docs)
    kwds = (
        kwd_df.groupby("keyword")
            .count()
            .query(f"doc_id > {min_thresh}")
            .query(f"doc_id < {max_thresh_int}")
            .index
    )

    # Go back and string match the keywords against all titles and abstracts.
    # Do this because RAKE gives us candidate keywords but does not assure us of their
    # locations. Only says keyword is present if it passes through RAKE.
    LOG.info("Going back through dataset to find keyword doc_id locations.")
    doc_to_kwds = ta.progress_apply(lambda x: [k for k in kwds if k in x])
    # doc_to_kwds = ta.progress_apply(lambda x: [k for k in kwds if k in x])
    dke = doc_to_kwds.explode().reset_index()
    dke.columns = ["doc_id", "keyword"]
    LOG.info("Filling years column.")
    doc_to_year = kwd_df.groupby("doc_id").agg(
        {"year": lambda x: x[0], "nasa_afil": lambda x: x[0]}
    )
    dke["rake_score"] = np.nan
    dke["keyword"] = dke["keyword"].astype(str)
    ys = doc_to_year.reindex(dke["doc_id"])["year"].tolist()
    nasa_afil = doc_to_year.reindex(dke["doc_id"])["nasa_afil"].tolist()
    LOG.info(f"Years with len: {len(ys)}")
    dke["year"] = ys
    dke["nasa_afil"] = nasa_afil
    LOG.info(f"dke with len: {len(dke)}")
    overlap_inds = pd.merge(dke.reset_index(), kwd_df, on=["doc_id", "keyword"])[
        "index"
    ]
    sdke = dke[~dke.index.isin(overlap_inds)]
    c_df = pd.concat([sdke, kwd_df]).sort_values("doc_id").reset_index(drop=True)
    na_ind = c_df["year"].isna()
    LOG.info(f"Remove {sum(na_ind)} keywords with NaN years.")
    c_df = c_df[~na_ind]
    c_df["year"] = c_df["year"].astype(int)
    return c_df


def stem_kwds(df):
    LOG.info("Creating keyword stems.")
    df = df.copy()
    unq_kwds = df["keyword"].astype(str).unique()
    p = PorterStemmer()
    kwd_to_stem = {kwd: p.stem(kwd) for kwd in tqdm(unq_kwds)}
    # Could also remove start with s here, also could resolve the
    df["stem"] = df["keyword"].progress_apply(lambda x: kwd_to_stem[x].lower().strip())
    return df


def binarize_years(df):
    LOG.info("Binarizing year columns.")
    df = df.copy()
    lb = LabelBinarizer()
    df['year'] = df['year'].astype(np.int16)
    year_binary = lb.fit_transform(df["year"])  # there should not be NAs.
    year_binary_df = pd.DataFrame(year_binary)
    year_binary_df.columns = lb.classes_
    # Write both to h5py, load in chunks and write concat to another h5py?
    df = pd.concat([df, year_binary_df], axis=1)
    return df


def stem_reduce(df, min_thresh):
    df = df.copy()
    LOG.info('Filtering down by keyword stems.')
    kwds_counts = df['stem'].value_counts()
    valid_stems = kwds_counts[kwds_counts > min_thresh].index
    s0 = df.shape[0]
    df = df.set_index('stem').loc[valid_stems, :].reset_index()
    s1 = df.shape[0]
    LOG.info(f'Removed {s0-s1} rows from dataframe.')
    return df


def get_stem_aggs(df):
    LOG.info("Aggregating by stems")
    df = df.copy()
    df["nasa_afil"] = df["nasa_afil"].apply(lambda x: 1 if x == "YES" else 0)
    years = np.sort(df["year"].unique())
    year_count_dict = {c: "sum" for c in years if not np.isnan(c)}
    df = df.groupby("stem").agg(
        {
            **{
                "rake_score": "mean",
                "doc_id": ["count", list],
                "keyword": lambda x: list(set(x)),
                "nasa_afil": lambda x: x.sum() / len(x),
            },
            **year_count_dict,
        }
    )
    df.columns = [f"{c}_{v}" for c, v in df.columns.values]
    df = df.rename(
        columns={"keyword_<lambda>": "keyword_list", "nasa_afil_<lambda>": "nasa_afil"}
    )
    return df


@click.option("--infile", type=Path)
@click.option("--outfile", type=Path)
@click.option("--out_years", type=Path)
@click.option("--min_thresh", type=int, default=0)
@click.option("--strategy", type=str, default="singlerank")
@click.option("--batch_size", type=int, default=1000)
@click.option("--n_process", type=int, default=1)
def main(infile, outfile, out_years, min_thresh, strategy, batch_size, n_process):
    """
    Get dataframe of keyword frequencies over the years
    """
    # TODO: this file above should go in size folder so only one to be changed with exp

    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    text = df["title"] + ". " + df["abstract"]
    text = text.apply(unescape).astype(str)
    text = text.apply(strip_tags).astype(str)
    strats = ["rake", "singlerank"]
    if strategy not in strats:
        raise ValueError(f"{strategy} not in {strats}.")
    if strategy == "rake":
        df["rake_kwds"] = get_keywords_from_text(text)
    elif strategy == "singlerank":
        df["rake_kwds"] = get_singlerank_kwds(text, batch_size, n_process)
    df = df.drop(
        [
            "arxiv_class",
            "alternate_bibcode",
            "keyword",
            "ack",
            "aff",
            "bibstem",
            "aff_id",
            "citation_count",
        ],
        axis=1,
    )
    df['title'] = df['title'].astype(str)
    df['abstract'] = df['abstract'].astype(str)
    df['year'] = df['year'].astype(int)
    kwd_df, year_counts = flatten_to_keywords(df, min_thresh)
    LOG.info(f"Writing out all keywords to {outfile}.")
    kwd_df.to_json(outfile, orient="records", lines=True)
    LOG.info(f"Writing year counts to {out_years}.")
    year_counts.to_csv(out_years)


if __name__ == "__main__":
    main()


