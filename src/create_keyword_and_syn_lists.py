import argparse
import logging
import os
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from pandarallel import pandarallel
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tqdm import tqdm
from memory_profiler import profile

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")
if "NB_WORKERS" in os.environ:
    NB_WORKERS = int(os.environ['NB_WORKERS'])
else:
    NB_WORKERS = cpu_count()
LOG.info(f'Using {NB_WORKERS} workers for pandarallel')

tqdm.pandas()
pandarallel.initialize(nb_workers=NB_WORKERS, progress_bar=True)


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
    # doc_to_kwds = ta.parallel_apply(lambda x: [k for k in kwds if k in x])
    doc_to_kwds = ta.progress_apply(lambda x: [k for k in kwds if k in x])
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


def stem_kwds(df):
    LOG.info("Creating keyword stems.")
    df = df.copy()
    unq_kwds = df["keyword"].astype(str).unique()
    p = PorterStemmer()
    kwd_to_stem = {kwd: p.stem(kwd) for kwd in tqdm(unq_kwds)}
    # Could also remove start with s here, also could resolve the
    df["stem"] = df["keyword"].progress_apply(lambda x: kwd_to_stem[x].lower().strip())
    return df


@profile
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


def norm_year_counts(df, year_counts):
    # Normalize columns by total count of year,
    # don't favor year with overall greater publication, do it by portion of total
    for year in year_counts.index:
        df[f"{year}_sum"] = df[f"{year}_sum"] / year_counts.loc[year, "count"]
    return df


def normalize(df, year_counts):
    df = df.copy()
    df = norm_year_counts(df, year_counts)
    sum_cols = [f"{year}_sum" for year in year_counts.index]
    scaler = MinMaxScaler()
    df.loc[:, sum_cols] = scaler.fit_transform(df.loc[:, sum_cols].T).T
    return df


def normalize_by_perc(df, year_counts):
    df = df.copy()
    df = norm_year_counts(df, year_counts)
    sum_cols = [f"{year}_sum" for year in year_counts.index]
    df.loc[:, sum_cols] = perc_change_from_baseline(df.loc[:, sum_cols].values)
    return df


def perc_change_from_baseline(x):
    def f(x):
        nzs = np.nonzero(x)[0]
        if len(nzs) is 0:
            vals = np.tile(np.nan, len(x))
        else:
            first_nz_i = nzs[0]
            val = x[first_nz_i]
            nvals = np.tile(np.nan, first_nz_i)
            fvals = np.tile(val, len(x) - first_nz_i)
            vals = np.concatenate((nvals, fvals))
        return vals

    baseline = np.apply_along_axis(f, 1, x)
    perc_vals = x / baseline
    return perc_vals


def dynamic_time_warping_sim(u, v):
    d, _, _, _ = dtw(u, v, dist=euclidean, w=5)
    # Above selections are somewhat arbitrary. Should determine more rigorously.
    return d


def flatten_to_keywords(df, min_thresh=5):
    allowed_db = "astronomy"
    LOG.info(f"Limiting to documents in database {allowed_db}")
    df = df[df["database"].apply(lambda x: allowed_db in x)]
    na_years = df["year"].isna()
    LOG.info(f"Remove {sum(na_years)} rows with NaN years.")
    df = df[~na_years]
    # kwd_df = (
    df = df.pipe(get_kwd_occurences, min_thresh)
    df = df.pipe(stem_kwds)
    df = df.pipe(stem_reduce, min_thresh)
    df = df.pipe(binarize_years)
    kwd_df = df.pipe(get_stem_aggs)
    # )
    year_counts = df["year"].value_counts().reset_index()
    year_counts.columns = ["year", "count"]
    kwd_df = kwd_df.reset_index()
    return kwd_df, year_counts


def main(infile: Path, out_dir: Path):
    assert not out_dir.exists(), LOG.exception(f"{out_dir} already exists.")
    out_dir.mkdir()

    # LOG.info(f"Reading keywords from {infile}.")
    # df = pd.read_json(infile, orient="records", lines=True)
    # df = df[df["database"].apply(lambda x: "astronomy" in x)].copy()
    # kwd_df = (
    #     df.pipe(get_kwd_occurences)
    #     .pipe(binarize_years)
    #     .pipe(stem_kwds)
    #     .pipe(get_stem_aggs)
    # )
    # LOG.info("Writing out all keywords.")
    # kwd_df.reset_index().to_json(
    #     out_dir / "all_keywords.jsonl", orient="records", lines=True
    # )
    outfile = out_dir / "all_keywords.jsonl"
    kwd_df = flatten_to_keywords(infile, outfile)

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
    # lim_kwd_norm_df = lim_kwd_df.pipe(normalize, year_counts)
    lim_kwd_norm_df = lim_kwd_df.pipe(normalize_by_perc_change, year_counts)
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
