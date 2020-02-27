import argparse
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import src.create_keyword_and_syn_lists as ck
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_resamples(df, min_thresh=100):
    allowed_db = "astronomy"  # TODO: Don't hardcode
    # TODO: some of this code duplicates the original code for flattening to kwds. Fix
    LOG.info(f"Limiting to documents in database {allowed_db}")
    df = df[df["database"].apply(lambda x: allowed_db in x)]
    na_years = df["year"].isna()
    LOG.info(f'Remove {sum(na_years)} rows with NaN years.')
    df = df[~na_years]
    boot_df = df.pipe(ck.get_kwd_occurences, min_thresh)
    n = 10 # TODO: Don't hardcode number of bootstraps
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input ADS records')
    parser.add_argument('o', help='output transformed bootstrap')
    parser.add_argument('y', help='output year counts')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.i, args.o, args.y)
