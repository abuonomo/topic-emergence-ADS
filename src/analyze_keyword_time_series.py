import argparse
import logging
from pathlib import Path

import bokeh.io
import ipyvolume as ipv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, ColumnDataSource, DEFAULT_TOOLS
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
from tqdm import tqdm
from tsfresh.feature_extraction import feature_calculators as fc
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def plot_time(v, show=False, lnr=False, size=(2, 2)):
    years = v.index.str.replace("_sum", "").astype(int)
    vals = v.values
    # plt.style.use("seaborn-dark")
    plt.figure(figsize=size)  # Must be square for tensorboard
    plt.plot(years, vals, color="black", linewidth=5)
    if lnr:
        l = linregress(years, vals)
        plt.plot(
            years,
            l.slope * years + l.intercept,
            linestyle=":",
            color="black",
            linewidth=2,
        )
    plt.xlabel("year")
    plt.ylabel("frequency")
    plt.title(v.name)
    fig = plt.gcf()
    if show:
        plt.show()
    return fig


def get_fig_images(df):
    images = []
    LOG.info("Getting images of time series.")
    for kwd in tqdm(df.index):
        fig = plot_time(df.loc[kwd], lnr=True)
        fig_img = figure_to_image(fig)
        images.append(fig_img)
    image_arr = np.stack(images)
    return image_arr


def test_kmeans_clusters(X):
    distortions = []
    K = range(1, 20)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(X)
        kmeans.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeans.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    # Plot the elbow
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()


def to_tboard(dtw_df, kmeans, images, lim):
    writer = SummaryWriter()
    meta = list(zip(dtw_df.index.tolist(), [str(l) for l in kmeans.labels_]))
    LOG.info("Writing to tensorboard.")
    writer.add_embedding(
        dtw_df.values[0:lim],
        metadata=meta[0:lim],
        label_img=images[0:lim],
        metadata_header=["keyword", "cluster"],
    )
    writer.close()
    LOG.info('Use "tensorboard --logdir runs" command to see visualization.')


def ts_to_tboard(normed_kwd_years):
    """Use tslearn to cluster timeseries"""
    ts_normed = to_time_series_dataset(normed_kwd_years.values)
    metric = "euclidean"
    LOG.info(f"Performing kmeans with metric {metric}.")
    km = TimeSeriesKMeans(n_clusters=6, metric=metric)
    km.fit(ts_normed)

    lim = 1000
    to_tboard(normed_kwd_years)
    images = get_fig_images(normed_kwd_years[0:lim])
    to_tboard(normed_kwd_years, km, images, lim)


def dtw_to_tboard(normed_kwd_years, dtw_df, c=6, lim=1000):
    """Use pre-computed dynamic time warp values and calculate kmeans."""
    # TODO: Used elbow to determine, but not being placed programmatically
    LOG.info(f"Performing kmeans with {c} clusters.")
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(dtw_df.values)

    images = get_fig_images(normed_kwd_years[0:lim])
    to_tboard(dtw_df, kmeans, images, lim)


def get_slopes(df):
    x = np.arange(0, df.shape[1])

    def f(y):
        l = linregress(x, y)
        s = l
        return s

    slopes = df.apply(f, axis=1)
    return slopes


def plot_slop_complex(se_df, viz_dir, x_measure="slope", y_measure="complexity"):
    output_file("slope_complex.html")
    se_df["log_count"] = np.log(se_df["count"])
    source = ColumnDataSource(se_df)
    tools = f"box_select,{DEFAULT_TOOLS}"
    LOG.info(f"Using tools {tools}")
    tooltips = [
        ("index", "$index"),
        ("keyword", "@keyword"),
    ]
    p1 = figure(
        title="keyword slopes versus complexity",
        tools=tools,
        tooltips=tooltips,
        x_axis_label=x_measure,
        y_axis_label=y_measure,
    )
    p1.circle(x_measure, y_measure, size="log_count", source=source, alpha=0.2)
    LOG.info("Using labels...")
    LOG.info(f"{se_df.columns}")
    p = p1
    out_bok = viz_dir / "slope_complex.html"
    LOG.info(f"Writing bokeh viz to {out_bok}")
    bokeh.io.save(p, viz_dir / "slope_complex.html")

    x = se_df["slope"].values
    y = se_df["complexity"].values
    z = se_df["count"].values.astype(np.float)
    ipv.scatter(x, y, z, size=1, marker="sphere")
    ipv.xlabel("slope")
    ipv.ylabel("complexity")
    ipv.zlabel("count")
    out_vol_viz = viz_dir / "slope_complex_count.html"
    LOG.info(f"Writing volume viz to {out_vol_viz}")
    ipv.save(out_vol_viz)


def filter_kwds(kwd_df, out_loc, threshold=50, score_thresh=1.3, hard_limit=10_000):
    LOG.info(f"Only getting keywords which occur in more than {threshold} docs.")
    lim_kwd_df = (
        kwd_df.query(f"doc_id_count > {threshold}")
        .query(f"rake_score_mean > {score_thresh}")
        .sort_values("rake_score_mean", ascending=False)
        .iloc[0:hard_limit]
    )
    LOG.info(f"Writing dataframe with size {lim_kwd_df.shape[0]} to {out_loc}.")
    lim_kwd_df.to_json(out_loc, orient="records", lines=True)
    return lim_kwd_df


def slope_count_complexity(lim_kwd_df, out_csv):
    only_years = lim_kwd_df.iloc[:, 5:]
    slopes_and_err = get_slopes(only_years)
    se_df = slopes_and_err.apply(pd.Series)
    se_df.columns = ["slope", "intercept", "r_value", "p_value", "std_err"]
    se_df["complexity"] = only_years.apply(lambda x: fc.cid_ce(x, False), axis=1)
    se_df["mean_change"] = only_years.apply(lambda x: fc.mean_change(x), axis=1)
    se_df["number_cwt_peaks"] = only_years.apply(
        lambda x: fc.number_cwt_peaks(x, 3), axis=1
    )
    # TODO: mean change per year, other tsfresh metrics
    se_df["keyword"] = lim_kwd_df["stem"]
    se_df["count"] = lim_kwd_df["doc_id_count"]
    LOG.info(f"Writing slope complexity data to {out_csv}")
    se_df.to_csv(out_csv)


def main(in_dir: Path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input txt file", type=Path)
    args = parser.parse_args()
    main(args.i)
