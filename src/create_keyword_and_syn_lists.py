import argparse
import json
import logging

import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from tqdm import tqdm

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


def main(infile, outfile, out_syn_file):
    LOG.info(f"Reading keywords from {infile}.")
    with open(infile, "r") as f0:
        rake_kwds = [json.loads(s) for s in tqdm(f0.readlines())]
    all_kwds = [(i, k[0], k[1]) for i, ks in enumerate(rake_kwds) for k in ks]
    kwd_df = pd.DataFrame(all_kwds)
    kwd_df.columns = ["doc_id", "keyword", "rake_score"]
    kwd_agg_df = kwd_df.groupby("keyword").agg(
        {"rake_score": "mean", "doc_id": "count"}
    )
    kwd_agg_df.columns = ["rake_score_mean", "doc_id_count"]

    # threshold = np.ceil(len(rake_kwds) / 200.0)  # TODO: determine this number
    threshold = 50
    score_thresh = 1.3
    hard_limit = 10_000
    LOG.info(f"Only getting keywords which occur in more than {threshold} docs.")
    # These limitation remove works before potential inclusion in synonym sets
    lim_kwd_agg_df = kwd_agg_df[kwd_agg_df["doc_id_count"] > threshold]
    lim_kwd_agg_df = lim_kwd_agg_df[kwd_agg_df["rake_score_mean"] > score_thresh]
    lim_kwd_agg_df = lim_kwd_agg_df.sort_values(
        "rake_score_mean", ascending=False
    ).iloc[0:hard_limit]
    LOG.info(f"Writing {lim_kwd_agg_df.shape[0]} keywords to {outfile}.")
    with open(outfile, "w") as f0:
        for kwd in tqdm(lim_kwd_agg_df.index):
            f0.write(kwd)
            f0.write("\n")

    LOG.info("Creating keyword stems.")
    p = PorterStemmer()
    tqdm.pandas()
    stem = lambda x: p.stem(x.name)
    lim_kwd_agg_df["stems"] = lim_kwd_agg_df.progress_apply(stem, axis=1)
    stem_groups = lim_kwd_agg_df.groupby("stems").groups
    lim_sg = {s: g for s, g in stem_groups.items() if len(g) > 1}
    LOG.info(f"Writing {len(lim_sg)} synonym sets to {out_syn_file}.")
    write_syn_file(out_syn_file, lim_sg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input json lines records")
    parser.add_argument("o", help="ouput text file with keywords")
    parser.add_argument("s", help="ouput text file with synonym sets")
    args = parser.parse_args()
    main(args.i, args.o, args.s)
