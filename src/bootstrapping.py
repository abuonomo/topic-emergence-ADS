import argparse
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import src.analyze_keyword_time_series as ak
import src.create_keyword_and_syn_lists as ck
import src.dtw_time_analysis as ad
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_resamples(df, min_thresh=100, n=10):
    allowed_db = "astronomy"  # TODO: Don't hardcode
    # TODO: some of this code duplicates the original code for flattening to kwds. Fix
    LOG.info(f"Limiting to documents in database {allowed_db}")
    df = df[df["database"].apply(lambda x: allowed_db in x)]
    na_years = df["year"].isna()
    LOG.info(f'Remove {sum(na_years)} rows with NaN years.')
    df = df[~na_years]
    boot_df = df.pipe(ck.get_kwd_occurences, min_thresh)
    kwd_df = boot_df.pipe(ck.binarize_years).pipe(ck.stem_kwds)
    LOG.info(f"Using {n} resamples for bootstrapping.")
    resampled_doc_ids = np.array([resample(kwd_df['doc_id'].unique()) for _ in range(n)])
    pbar = tqdm(resampled_doc_ids)
    kwd_df_resamples = []
    year_counts_resamples = []
    for tmp_ind in pbar:
        kwd_df_boot = pd.concat([kwd_df[kwd_df['doc_id'] == i] for i in tmp_ind])
        kwd_df_boot = kwd_df_boot.pipe(ck.get_stem_aggs)
        kwd_df_resamples.append(kwd_df_boot)

        year_counts = df.loc[tmp_ind, "year"].value_counts().reset_index()
        year_counts.columns = ["year", "count"]
        year_counts_resamples.append(year_counts)

    # need resampled year count totals
    return kwd_df_resamples, year_counts_resamples


def main(infile):  #, outfile, out_years, min_thresh=100):
    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    kwd_df_resamples, year_counts_resamples = get_resamples(df, min_thresh=100, n=5)
    # TODO: feed bootstraps into full pipeline to create all tsfresh metrics, this time with a distribution. How to visualize with distribution?
    # TODO: get time estimate on how long that would take, knowing how fast one is shoudl be a good approximation.
    for kwd_df, year_counts in zip(kwd_df_resamples, year_counts_resamples):
        kwd_df = kwd_df.reset_index()
        import ipdb; ipdb.set_trace()
        lim_kwd_df = ak.filter_kwds(
            kwd_df, threshold=20, score_thresh=0.05, hard_limit=10_000
        )
        year_counts_tmp = year_counts.sort_values("year").set_index("year")
        normed_df = ck.normalize_by_perc(lim_kwd_df, year_counts_tmp)
        features = ak.slope_count_complexity(normed_df)
        normed_df_years = normed_df.set_index("stem").iloc[:, 5:]
        dtw_df = ad.dtw_kwds(normed_df_years)
        kmeans = ak.dtw_to_tboard(normed_df, dtw_df, c=7)  # c taken from elbow viz
        dtw_man = ak.dtw_to_manifold(dtw_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input ADS records')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.i)
