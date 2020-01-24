import argparse
import json
import logging
from pathlib import Path

from dtw import dtw
import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from tqdm import tqdm
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")


def write_syn_file(out_syn_file, lim_sg):
    pbar = tqdm(enumerate(lim_sg.items()), total=len(lim_sg))
    with open(out_syn_file, "w") as f0:
        for i, (s, g) in pbar:
            for kwd in g:
                f0.write(kwd)
                f0.write("\n")
            if i != len(lim_sg) - 1:
                f0.write("---")
                f0.write("\n")


def get_kwd_occurences(df):
    LOG.info("Flattening along document rake_kwds.")
    df = df.copy()
    df = df.applymap(lambda x: np.nan if x is None else x)
    kwd_inds = df["rake_kwds"].apply(lambda x: type(x) is list)
    rake_kwds = df["rake_kwds"][kwd_inds]
    all_kwds = [(i, k[0], k[1]) for i, r in zip(rake_kwds.index, rake_kwds) for k in r]
    kwd_df = pd.DataFrame(all_kwds)
    kwd_df.columns = ["doc_id", "keyword", "rake_score"]
    kwd_df["year"] = df.loc[kwd_df["doc_id"], "year"].tolist()
    return kwd_df


def binarize_years(df):
    LOG.info("Binarizing year columns.")
    df = df.copy()
    lb = LabelBinarizer()
    year_binary = lb.fit_transform(df["year"])
    year_binary_df = pd.DataFrame(year_binary)
    year_binary_df.columns = lb.classes_
    df = pd.concat([df, year_binary_df], axis=1)
    return df


def stem_kwds(df):
    LOG.info("Creating keyword stems.")
    df = df.copy()
    unq_kwds = df["keyword"].unique()
    p = PorterStemmer()
    kwd_to_stem = {kwd: p.stem(kwd) for kwd in tqdm(unq_kwds)}
    tqdm.pandas()
    df["stem"] = df["keyword"].progress_apply(lambda x: kwd_to_stem[x])
    return df


def get_stem_aggs(df):
    df = df.copy()
    years = np.sort(df["year"].unique())
    year_count_dict = {c: "sum" for c in years}
    df = df.groupby("stem").agg(
        {**{"rake_score": "mean", "doc_id": "count"}, **year_count_dict}
    )
    cols = ["rake_score_mean", "doc_id_count"] + [f"{s}_sum" for s in years]
    df.columns = cols
    return df


def normalize(df, year_counts):
    df = df.copy()
    scaler = StandardScaler()
    # Normalize columns by total count of year,
    # don't favor year with overall greater publication, do it by portion of total
    for year in year_counts.index:
        df[f"{year}_sum"] = df[f"{year}_sum"] / year_counts[year]
    sum_cols = [f"{year}_sum" for year in year_counts.index]
    df.loc[:, sum_cols] = scaler.fit_transform(df.loc[:, sum_cols].T).T
    return df


def dynamic_time_warping_sim(u, v):
    d, _, _, _ = dtw(u, v, dist=euclidean, w=5)
    # Above selection are somewhat arbitrary. Should determine more rigorously.
    return d


def main(infile: Path, out_dir: Path):
    assert not out_dir.exists(), LOG.exception(f'{out_dir} already exists.')
    out_dir.mkdir()

    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    kwd_df = (
        df.pipe(get_kwd_occurences)
        .pipe(binarize_years)
        .pipe(stem_kwds)
        .pipe(get_stem_aggs)
    )
    kwd_df.to_json(out_dir / "all_keywords.jsonl", orient="records", lines=True)

    # threshold = np.ceil(len(rake_kwds) / 200.0)  # TODO: determine this number
    threshold = 50
    score_thresh = 1.3
    hard_limit = 10_000
    LOG.info(f"Only getting keywords which occur in more than {threshold} docs.")
    lim_kwd_df = (
        kwd_df.query(f"doc_id_count > {threshold}")
        .query(f"rake_score_mean > {score_thresh}")
        .sort_values("rake_score_mean", ascending=False)
        .iloc[0:hard_limit]
    )
    year_counts = df["year"].value_counts()
    lim_kwd_norm_df = lim_kwd_df.pipe(normalize, year_counts)
    sum_cols = [f"{year}_sum" for year in np.sort(year_counts.index)]
    normed_kwd_years = lim_kwd_norm_df[sum_cols]
    norm_loc = out_dir / "lim_normed_keyword_stems.jsonl"
    LOG.info(f"Writing normalized keywords years to {norm_loc}.")
    normed_kwd_years.to_csv(norm_loc)

    out_kwds = out_dir / "kwds_for_semnet.txt"
    LOG.info(f"Writing {lim_kwd_df.shape[0]} keywords to {out_kwds}.")
    with open(out_kwds, "w") as f0:
        for kwd in tqdm(normed_kwd_years.index):
            f0.write(kwd)
            f0.write("\n")

    LOG.info("Computing dynamic time warping between keywords.")
    dtw_dists = pdist(normed_kwd_years, dynamic_time_warping_sim)
    dtw_df = pd.DataFrame(squareform(dtw_dists))
    dtw_df.columns = normed_kwd_years.index
    dtw_df.index = normed_kwd_years.index
    dtw_loc = out_dir / "dynamic_time_warp_distances.csv"
    LOG.info(f"Outputting dynamic time warps to {dtw_loc}.")
    dtw_df.to_csv(dtw_loc)

    stem_groups = lim_kwd_df.groupby("stem").groups
    lim_sg = {s: g for s, g in stem_groups.items() if len(g) > 1}
    out_syn_file = out_dir / "syns_for_semnet.txt"
    LOG.info(f"Writing {len(lim_sg)} synonym sets to {out_syn_file}.")
    write_syn_file(out_syn_file, lim_sg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input json lines records", type=Path)
    parser.add_argument("o", help="ouput dir for results", type=Path)
    args = parser.parse_args()
    main(args.i, args.o)
