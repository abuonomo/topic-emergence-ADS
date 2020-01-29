import argparse
import logging
from bokeh.plotting import figure, output_file, show, ColumnDataSource, DEFAULT_TOOLS
from bokeh.models import HoverTool, LabelSet
from bokeh.layouts import gridplot
import matplotlib.pyplot as plt
import mpld3
import pandas as pd
from pathlib import Path
import ipyvolume as ipv
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tsfresh import extract_features
from tsfresh.feature_extraction import feature_calculators as fc
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
from collections import OrderedDict


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


def ts_to_tboard(normed_kwd_years):
    """Use tslearn to cluster timeseries"""
    ts_normed = to_time_series_dataset(normed_kwd_years.values)
    metric = "euclidean"
    LOG.info(f"Performing kmeans with metric {metric}.")
    km = TimeSeriesKMeans(n_clusters=6, metric=metric)
    km.fit(ts_normed)

    lim = 1000
    writer = SummaryWriter()
    images = get_fig_images(normed_kwd_years[0:lim])
    meta = list(zip(normed_kwd_years.index.tolist(), [str(l) for l in km.labels_]))
    LOG.info("Writing to tensorboard.")
    writer.add_embedding(
        normed_kwd_years.values[0:lim],
        metadata=meta[0:lim],
        label_img=images[0:lim],
        metadata_header=["keyword", "cluster"],
    )
    writer.close()


def dtw_to_tboard(normed_kwd_years, dtw_df):
    """Use pre-computed dynamic time warp values and calculate kmeans."""
    # TODO: Used elbow to determine, but not being placed programmatically
    c = 6
    LOG.info(f"Performing kmeans with {c} clusters.")
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(dtw_df.values)

    lim = 1000
    writer = SummaryWriter()
    images = get_fig_images(normed_kwd_years[0:lim])
    meta = list(zip(dtw_df.index.tolist(), [str(l) for l in kmeans.labels_]))
    LOG.info("Writing to tensorboard.")
    writer.add_embedding(
        dtw_df.values[0:lim],
        metadata=meta[0:lim],
        label_img=images[0:lim],
        metadata_header=["keyword", "cluster"],
    )
    writer.close()


def get_slopes(df):
    x = np.arange(0, df.shape[1])

    def f(y):
        l = linregress(x, y)
        s = l
        return s

    slopes = df.apply(f, axis=1)
    return slopes


def plot_slop_complex(se_df, out_viz):
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
        x_axis_label='slope',
        y_axis_label='complexity',
        # x_axis_type="log"
        # y_axis_type="log",
    )
    # p2 = figure(
    #     title="keyword slopes versus count",
    #     x_range=p1.x_range,
    #     tools=tools,
    #     tooltips=tooltips,
    # )
    p1.circle("slope", "complexity", size="log_count", source=source, alpha=0.2)
    # p2.scatter("slope", "count", source=source, alpha=0.2)
    LOG.info("Using labels...")
    LOG.info(f"{se_df.columns}")
    p = p1
    # p = gridplot([[p1, p2]], toolbar_location="below")
    show(p)

    x = se_df["slope"].values
    y = se_df["complexity"].values
    z = se_df["count"].values.astype(np.float)
    ipv.scatter(x, y, z, size=1, marker="sphere")
    ipv.xlabel("slope")
    ipv.ylabel("complexity")
    ipv.zlabel("count")
    ipv.save(out_viz)


def filter_kwds(kwd_df, out_loc, threshold=50, score_thresh=1.3, hard_limit=10_000):
    LOG.info(f"Only getting keywords which occur in more than {threshold} docs.")
    lim_kwd_df = (
        kwd_df.query(f"doc_id_count > {threshold}")
        .query(f"rake_score_mean > {score_thresh}")
        .sort_values("rake_score_mean", ascending=False)
        .iloc[0:hard_limit]
    )
    LOG.info(f"Writing to {out_loc}.")
    lim_kwd_df.to_csv(out_loc)
    return lim_kwd_df


def slope_count_complexity(lim_kwd_df, out_csv):
    only_years = lim_kwd_df.iloc[:, 3:]
    slopes_and_err = get_slopes(only_years)
    se_df = slopes_and_err.apply(pd.Series)
    se_df.columns = ["slope", "intercept", "r_value", "p_value", "std_err"]
    se_df["complexity"] = only_years.apply(lambda x: fc.cid_ce(x, False), axis=1)
    se_df["keyword"] = lim_kwd_df["stem"]
    se_df["count"] = lim_kwd_df["doc_id_count"]
    LOG.info(f'Writing slope complexity data to {out_csv}')
    se_df.to_csv(out_csv)


def main(in_dir: Path):
    # dtw_loc = in_dir / "dynamic_time_warp_distances.csv"
    # LOG.info(f"Reading dynamic time warps from {dtw_loc}.")
    # dtw_df = pd.read_csv(in_dir / "dynamic_time_warp_distances.csv", index_col=0)

    norm_loc = in_dir / "lim_normed_keyword_stems.jsonl"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwd_years = pd.read_csv(norm_loc, index_col=0)
    lim = 1000
    writer = SummaryWriter()
    slopes_and_err = get_slopes(normed_kwd_years)
    se_df = slopes_and_err.apply(pd.Series)
    se_df.columns = ["slope", "intercept", "r_value", "p_value", "std_err"]
    slopes = se_df["slope"]
    se_df["complexity"] = normed_kwd_years.apply(lambda x: fc.cid_ce(x, False), axis=1)
    se_df.reset_index()
    # fig, ax = plt.subplots()
    # ax.plot(slopes, changes, 'o', alpha=0.2)
    # ax.set_xlabel("Linear Regression Slopes")
    # ax.set_ylabel("Complexity")
    # ax.set_title("Keyword Slope Versus Complexity")
    # fig.show()
    # labels = slopes.index.to_list()
    # tooltip = mpld3.plugins.PointLabelTooltip(ax, labels=labels)
    # mpld3.plugins.connect(fig, tooltip)
    # mpld3.save_html(fig, 'tmp.html')
    sm_df = pd.DataFrame([slopes, changes]).T
    meta = list(
        zip(
            normed_kwd_years.index.tolist(),
            [str(l) for l in np.arange(0, normed_kwd_years.shape[0])],
        )
    )
    sm_df.columns = ["slope", "complexity"]
    ind_ord = slopes.sort_values(ascending=False).index
    images = get_fig_images(normed_kwd_years.loc[ind_ord][0:lim])
    z = zip(normed_kwd_years.loc[ind_ord].index[0:lim], images[0:lim])
    writer.add_embedding(
        sm_df.values[0:lim],
        metadata=meta[0:lim],
        label_img=images[0:lim],
        metadata_header=["keyword", "cluster"],
    )
    [writer.add_image(n, i) for n, i in z]
    writer.close()
    LOG.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input txt file", type=Path)
    args = parser.parse_args()
    main(args.i)
