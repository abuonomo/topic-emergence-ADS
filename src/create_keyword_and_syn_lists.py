import argparse
import logging
import os
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import spacy
from pandarallel import pandarallel
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.extract_keywords import flatten_to_keywords

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
