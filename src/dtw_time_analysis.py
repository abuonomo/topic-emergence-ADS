import argparse
import logging
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import tslearn.metrics as tsm
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, pdist, squareform
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def dynamic_time_warping_sim(u, v, w):
    d, _, _, _ = dtw(u, v, dist=euclidean, w=w)
    # Above selections are somewhat arbitrary. Should determine more rigorously.
    return d


def fast_sim(u, v, w):
    d, _ = fastdtw(u, v, radius=w, dist=euclidean)
    return d


def tsdtw(u, v, w):
    d = tsm.dtw(u, v, global_constraint="sakoe_chiba", sakoe_chiba_radius=w)
    return d


def plot_dtw_times(normed_kwd_years, s=25, top=1000, window=1):
    times = []
    sizes = list(range(0, top, s))
    sizes_pbar = tqdm(sizes)
    tt = None
    # dtw_sim = lambda u, v: fast_sim(u, v, w=window)
    dtw_sim = lambda u, v: tsdtw(u, v, w=window)
    for i in sizes_pbar:
        sizes_pbar.set_description(f"Size: {i} | {i - s} time: {tt}")
        t1 = time()
        dtw_dists = pdist(normed_kwd_years.iloc[0:i], dtw_sim)
        t2 = time()
        tt = t2 - t1
        times.append(tt)
    plt.plot(sizes, times)
    plt.ylabel("seconds")
    plt.xlabel("number of keywords")
    plt.title(f"Dynamic Time Warp Speed | Window: {window}")
    plt.show()
    return sizes, times


def dtw_kwds(normed_kwd_years, dtw_loc):
    LOG.info("Computing dynamic time warping between keywords.")
    window = 1
    LOG.info(f"window: {window}.")
    dtw_sim = lambda u, v: tsdtw(u, v, w=window)
    dtw_dists = pdist(normed_kwd_years, dtw_sim)
    dtw_df = pd.DataFrame(squareform(dtw_dists))
    dtw_df.columns = normed_kwd_years.index
    dtw_df.index = normed_kwd_years.index
    LOG.info(f"Outputting dynamic time warps to {dtw_loc}.")
    dtw_df.to_csv(dtw_loc)


def main(data_dir: Path):
    norm_loc = data_dir / "lim_normed_keyword_stems.jsonl"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwd_years = pd.read_csv(norm_loc, index_col=0)
    dtw_kwds(normed_kwd_years, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dynamic time warping")
    parser.add_argument("d", help="dir for input and output of keyword data", type=Path)
    args = parser.parse_args()
    main(args.d)
