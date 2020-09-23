import argparse
import logging
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import tslearn.metrics as tsm
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from yellowbrick.features import Manifold
from yellowbrick.cluster import KElbowVisualizer


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


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


def dtw_to_manifold(dtw_df, out_plot=None):
    LOG.info("Computing tsne manifold.")
    viz = Manifold(manifold="tsne")
    # Will throw an error if any of the topics have identical time series
    dtw_man = viz.fit_transform(dtw_df)  # Fit the data to the visualizer
    if out_plot is not None:
        LOG.info(f"Writing tsne manifold plot to {out_plot}.")
        viz.show(out_plot)  # Finalize and render the figure
    return dtw_man


def dtw_to_tboard(dtw_df, c=6) -> KMeans:
    """Use pre-computed dynamic time warp values and calculate kmeans."""
    # TODO: Used elbow to determine, but not being placed programmatically
    LOG.info(f"Performing kmeans with {c} clusters.")
    m = KMeans(n_clusters=c)
    m.fit(dtw_df.values)
    return m


def tsdtw(u, v, w):
    d = tsm.dtw(u, v, global_constraint="sakoe_chiba", sakoe_chiba_radius=w)
    return d


def plot_dtw_times(normed_kwd_years, s=25, top=1000, window=1):
    times = []
    sizes = list(range(0, top, s))
    sizes_pbar = tqdm(sizes)
    tt = None
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


def dtw_kwds(normed_kwd_years):
    LOG.info("Computing dynamic time warping between keywords.")
    window = 1
    LOG.info(f"window: {window}.")
    scaler = StandardScaler()
    znormed_kwds = scaler.fit_transform(normed_kwd_years.fillna(0).T).T
    dtw_sim = lambda u, v: tsdtw(u, v, w=window)
    dtw_dists = pdist(znormed_kwds, dtw_sim)
    dtw_df = pd.DataFrame(squareform(dtw_dists))
    dtw_df.columns = normed_kwd_years.index
    dtw_df.index = normed_kwd_years.index
    return dtw_df


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
