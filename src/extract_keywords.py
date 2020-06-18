import json
import ast
import os
from html import unescape
from html.parser import HTMLParser

import RAKE
import click
import logging
import numpy as np
import pandas as pd
import spacy
from nltk import PorterStemmer
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.lang.en import STOP_WORDS
from spacy.util import compile_infix_regex
from textacy.ke import textrank
from tqdm import tqdm
from typing import List

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")

# modify tokenizer infix patterns to not split on hyphen
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # EDIT: commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)
infix_re = compile_infix_regex(infixes)
NLP.tokenizer.infix_finditer = infix_re.finditer

tqdm.pandas()


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
            normalize=None,
            topn=999,  # This could technically cause issues with a huge abstract
            window_size=10,
            edge_weighting="count",
            position_bias=False,
        )
        text = doc.text
        t = []
        for i, (k, v) in enumerate(kwds):
            k_inds = [(i, j) for j in range(len(text)) if text.startswith(k, j)]
            t = t + k_inds
        st = sorted(t, key=lambda x: x[1])
        kwds_sorted = [kwds[i[0]] for i in st]
        kwd_lists.append(kwds_sorted)
        pbar.update(1)
    return kwd_lists


def flatten_to_keywords(df, min_thresh=5, max_thresh=1):
    df = df.pipe(get_kwd_occurences, min_thresh, max_thresh)
    # TODO: include gensim process from notebook arb-16?
    df = df.pipe(stem_kwds)
    df = df.pipe(stem_reduce, min_thresh)
    df = df.pipe(binarize_years)
    kwd_df = df.pipe(get_stem_aggs)
    kwd_df = kwd_df.reset_index()
    return kwd_df


def get_kwd_occurences(df, min_thresh=5, max_thresh=0.7):
    LOG.info("Flattening along document kwds.")
    df = df.copy()
    df = df.applymap(lambda x: np.nan if x is None else x)
    kwd_inds = df["kwds"].apply(lambda x: type(x) is list)
    kwds = df["kwds"][kwd_inds]
    all_kwds = [(i, k[0], k[1]) for i, r in zip(kwds.index, kwds) for k in r]

    kwd_df = pd.DataFrame(all_kwds)
    kwd_df.columns = ["doc_id", "keyword", "score"]
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
    n_removed = kwd_df.shape[0] - len(kwds)
    LOG.info(
        f"Removed {n_removed} keywords which occur less "
        f"than {min_thresh} times or in more than {max_thresh} of corpus"
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
    dke["score"] = np.nan
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
    # df = df.copy()
    unq_kwds = df["keyword"].astype(str).unique()
    p = PorterStemmer()
    kwd_to_stem = {kwd: p.stem(kwd) for kwd in tqdm(unq_kwds)}
    # Could also remove start with s here, also could resolve the
    df["stem"] = df["keyword"].progress_apply(lambda x: kwd_to_stem[x].lower().strip())
    return df


def binarize_years(df):
    LOG.info("Binarizing year columns.")
    # df = df.copy()
    lb = LabelBinarizer()
    df["year"] = df["year"].astype(np.int16)
    year_binary = lb.fit_transform(df["year"])  # there should not be NAs.
    year_binary_df = pd.DataFrame(year_binary)
    year_binary_df.index = df.index
    year_binary_df.columns = lb.classes_
    # Write both to h5py, load in chunks and write concat to another h5py?
    df = pd.concat([df, year_binary_df], axis=1)
    return df


def stem_reduce(df, min_thresh):
    # df = df.copy()
    LOG.info("Filtering down by keyword stems.")
    kwds_counts = df["stem"].value_counts()
    valid_stems = kwds_counts[kwds_counts > min_thresh].index
    s0 = df.shape[0]
    df = df.set_index("stem").loc[valid_stems, :].reset_index()
    s1 = df.shape[0]
    LOG.info(f"Removed {s0-s1} rows from dataframe.")
    return df


def get_stem_aggs(df):
    LOG.info("Aggregating by stems")
    # df = df.copy()
    df["nasa_afil"] = df["nasa_afil"].apply(lambda x: 1 if x == "YES" else 0)
    years = np.sort(df["year"].unique())
    year_count_dict = {c: "sum" for c in years if not np.isnan(c)}
    df = df.groupby("stem").agg(
        {
            **{
                "score": "mean",
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


def is_nu_like(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def filter_kwds_inner(kwd_df, threshold=50, score_thresh=1.3, hard_limit=10_000):
    LOG.info(f"Only getting keywords which occur in more than {threshold} docs.")
    lim_kwd_df = (
        kwd_df.query(f"doc_id_count > {threshold}")
        .query(f"score_mean > {score_thresh}")
        .sort_values("score_mean", ascending=False)
        .iloc[0:hard_limit]
    )
    tdf = lim_kwd_df.drop(
        lim_kwd_df.index[lim_kwd_df["stem"].apply(lambda x: len(x.strip()) == 1)]
    )
    tdf = tdf.drop(tdf.index[tdf["stem"].apply(is_nu_like)])
    return tdf


@click.group()
def cli():
    pass


@cli.command()
@click.option("--infile", type=Path)
@click.option("--out_loc", type=Path)
@click.option("--year_count_loc", type=Path)
@click.option("--threshold", type=int)
@click.option("--score_thresh", type=float)
@click.option("--hard_limit", type=int)
@click.option("--year_min", type=int, default=0)
@click.option("--drop_feature_loc", type=Path, default=None)
def filter_kwds(
    infile,
    out_loc,
    year_count_loc,
    threshold,
    score_thresh,
    hard_limit,
    year_min=0,
    drop_feature_loc=None,
):
    """
    Filter keywords by total frequency and score. Also provide hard limit.
    """
    LOG.info(f"Reading from {infile}")
    df = pd.read_json(infile, orient="records", lines=True)
    LOG.info(f"Reading year counts from {year_count_loc}.")

    year_count_df = pd.read_csv(year_count_loc, index_col=0)
    years = year_count_df["year"].sort_values().values

    dropycols = [f"{y}_sum" for y in years if y < year_min]
    df = df.drop(dropycols, axis=1)

    if drop_feature_loc is not None:
        with open(drop_feature_loc, 'r') as f0:
            drop_features = ast.literal_eval(f0.read())
    import ipdb; ipdb.set_trace()

    lim_kwd_df = filter_kwds_inner(df, threshold, score_thresh, hard_limit)
    LOG.info(f"Writing dataframe with size {lim_kwd_df.shape[0]} to {out_loc}.")
    lim_kwd_df.to_json(out_loc, orient="records", lines=True)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--out_years", type=Path)
@click.option("--out_tokens", type=Path)
@click.option("--strategy", type=str, default="singlerank")
@click.option("--batch_size", type=int, default=1000)
@click.option("--n_process", type=int, default=1)
def main(infile, out_years, out_tokens, strategy, batch_size, n_process):
    """
    Get dataframe of keyword frequencies over the years
    """
    # TODO: this file above should go in size folder so only one to be changed with exp

    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
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
    df["title"] = df["title"].astype(str)
    df["abstract"] = df["abstract"].astype(str)
    df["year"] = df["year"].astype(int)

    year_counts = df["year"].value_counts().reset_index()
    year_counts.columns = ["year", "count"]
    LOG.info(f"Writing year counts to {out_years}.")
    year_counts.to_csv(out_years)

    text = df["title"] + ". " + df["abstract"]
    text = text.apply(unescape).astype(str)
    text = text.apply(strip_tags).astype(str)

    strats = ["rake", "singlerank"]
    if strategy not in strats:
        raise ValueError(f"{strategy} not in {strats}.")
    if strategy == "rake":
        df["kwds"] = get_keywords_from_text(text)
    elif strategy == "singlerank":
        df["kwds"] = get_singlerank_kwds(text, batch_size, n_process)

    tokens = df["kwds"].tolist()
    LOG.info(f"Writing tokens to {out_tokens}")
    with open(out_tokens, "w") as f0:
        for doc_toks in tokens:
            f0.write(json.dumps(doc_toks))
            f0.write("\n")


@cli.command()
@click.option("--infile", type=Path)
@click.option("--in_kwd_lists", type=Path)
@click.option("--outfile", type=Path)
@click.option("--min_thresh", type=int, default=0)
@click.option("--max_thresh", type=float, default=1)
def aggregate_kwds(infile, in_kwd_lists, outfile, min_thresh, max_thresh):
    df = pd.read_json(infile, orient="records", lines=True)
    with open(in_kwd_lists, "r") as f0:
        kwds = [json.loads(l) for l in f0.read().splitlines()]
    df["kwds"] = kwds
    kwd_df = flatten_to_keywords(df, min_thresh, max_thresh)
    LOG.info(f"Writing out all keywords to {outfile}.")
    kwd_df.to_json(outfile, orient="records", lines=True)


if __name__ == "__main__":
    cli()
