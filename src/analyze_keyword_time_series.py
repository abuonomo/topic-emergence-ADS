import argparse
import logging
from pathlib import Path

import bokeh.io
import ipyvolume as ipv
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, output_file, ColumnDataSource, DEFAULT_TOOLS
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
from tqdm import tqdm
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import feature_calculators as fc
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import Manifold

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def plot_time(v, show=False, lnr=False, size=(2, 2)):
    years = v.index.str.replace("_sum", "").astype(int)
    vals = v.values
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


def to_tboard(dtw_df, m, images, lim):
    writer = SummaryWriter()
    meta = list(zip(dtw_df.index.tolist(), [str(l) for l in m.labels_]))
    LOG.info("Writing to tensorboard.")
    writer.add_embedding(
        dtw_df.values[0:lim],
        metadata=meta[0:lim],
        label_img=images[0:lim],
        metadata_header=["keyword", "cluster"],
    )
    writer.close()
    LOG.info('Use "tensorboard --logdir runs" command to see visualization.')


def yellow_plot_kmd(X, out_plot, c_min=2, c_max=20):
    LOG.info(f"Trying kmeans n_clusters from {c_min} to {c_max}")
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(c_min, c_max))
    visualizer.fit(X)  # Fit the data to the visualizer
    LOG.info(f"Writing elbow to {out_plot}.")
    visualizer.show(out_plot)


def plot_kmeans_distortions(X, out_plot, c_min=2, c_max=20):
    distortions = []
    K = range(c_min, c_max)
    LOG.info(f"Trying kmeans n_clusters from {c_min} to {c_max}")
    for k in tqdm(K):
        kmeans = KMeans(n_clusters=k).fit(X)
        kmeans.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeans.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    LOG.info(f"Writing kmeans elbox plot to {out_plot}")
    plt.savefig(out_plot)
    plt.clf()


def gm_bics(dtw_df, c_min=2, c_max=20):
    bics = []
    LOG.info(f"Trying Gaussian Mixture n_clusters from {c_min} to {c_max}")
    for c in tqdm(range(c_min, c_max)):
        m = GaussianMixture(n_components=c)
        m.fit(dtw_df.values)
        b = m.bic(dtw_df.values)
        bics.append(b)
    return bics


def plot_gm_bics(dtw_df, out_plot, c_min=2, c_max=20):
    bics = gm_bics(dtw_df, c_min, c_max)
    plt.plot(range(c_min, c_max), bics)
    plt.xlabel("number of clusters")
    plt.ylabel("Bayesian Information Criteria (BIC)")
    plt.title("Gaussian Mixture BIC by Number of Cluster")
    LOG.info(f"Writing BIC plot to {out_plot}")
    plt.savefig(out_plot)
    plt.clf()


def dtw_to_manifold(dtw_df, out_plot=None):
    LOG.info("Computing tsne manifold.")
    viz = Manifold(manifold="tsne")
    dtw_man = viz.fit_transform(dtw_df)  # Fit the data to the visualizer
    if out_plot is not None:
        LOG.info(f"Writing tsne manifold plot to {out_plot}.")
        viz.show(out_plot)  # Finalize and render the figure
    return dtw_man


def dtw_to_tboard(normed_kwd_years, dtw_df, c=6, lim=1000):
    """Use pre-computed dynamic time warp values and calculate kmeans."""
    # TODO: Used elbow to determine, but not being placed programmatically
    LOG.info(f"Performing kmeans with {c} clusters.")
    m = KMeans(n_clusters=c)
    m.fit(dtw_df.values)
    return m


def get_slopes(df):
    x = np.arange(0, df.shape[1])

    def f(y):
        ind = ~y.isna()
        l = linregress(x[ind], y[ind])
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

    # TODO: this will be broken with new time series feature extract
    # column names have changed.
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


def cagr(x):
    x = x.copy()
    first_nonzero_index = np.nonzero(x.values)[0][0]
    x = x[first_nonzero_index:]  # Not valid if starts with 0. Becomes inf
    x = x[~x.isna()]  # For normalized time series, NaNs before any occurrence of kwd
    ys = [int(s.split('_')[0]) for s in x.index]
    period = max(ys) - min(ys)
    return (x[-1] / x[0]) ** (1 / period) - 1


def slope_count_complexity(lim_kwd_df, overall_affil, years):
    year_cols = [f'{y}_sum' for y in years]
    only_years = lim_kwd_df.loc[:, year_cols]
    # TODO: using an index by number here is quite inflexible to change. Fix it.

    f2 = lambda x: fc.mean_change(x[~x.isna()])
    trans_t = (
        only_years.reset_index()
        .melt(id_vars=["index"])
        .sort_values(["index", "variable"])
    )
    trans_t["year"] = trans_t["variable"].apply(lambda x: int(x[0:4]))
    trans_t = trans_t.drop(columns=["variable"])
    features = extract_features(
        trans_t.fillna(0), column_id="index", column_sort="year"
    )
    features["count"] = lim_kwd_df["doc_id_count"]
    features["stem"] = lim_kwd_df["stem"]
    features["nasa_afil"] = lim_kwd_df["nasa_afil"]
    features["norm_nasa_afil"] = lim_kwd_df["nasa_afil"] / overall_affil

    up_ind = features["norm_nasa_afil"] >= 1
    down_ind = features["norm_nasa_afil"] <= 1

    sg1 = minmax_scale(features["norm_nasa_afil"][up_ind], feature_range=(0, 100))
    sl1 = minmax_scale(features["norm_nasa_afil"][down_ind], feature_range=(0, 1))
    features.loc[up_ind, "norm_nasa_afil"] = sg1
    features.loc[down_ind, "norm_nasa_afil"] = sl1

    features["mean_change_nan_before_exist"] = only_years.apply(f2, axis=1)
    import ipdb; ipdb.set_trace()
    features["cagr"] = only_years.apply(cagr, axis=1)
    features["score_mean"] = lim_kwd_df["score_mean"]
    return features


def main(in_dir: Path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input txt file", type=Path)
    args = parser.parse_args()
    main(args.i)
