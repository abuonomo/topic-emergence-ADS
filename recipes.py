import logging
from pathlib import Path

import click
import joblib
import pandas as pd
from scipy.io import mmwrite

from src.analyze_keyword_time_series import (
    slope_count_complexity,
    plot_slop_complex,
    dtw_to_manifold,
    yellow_plot_kmd,
    plot_time,
    filter_kwds,
    dtw_to_tboard,
)
from src.create_keyword_and_syn_lists import (
    flatten_to_keywords,
    normalize_by_perc,
)
from src.dtw_time_analysis import dtw_kwds
from src.topic_modeling import (
    topic_model_viz,
    get_doc_len_from_file,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

P_JOURNALS = ["Natu", 'Sci.']


@click.group()
def cli():
    pass


@cli.command()
@click.option("--infile", type=Path)
@click.option("--outfile", type=Path)
@click.option("--out_years", type=Path)
@click.option("--min_thresh", type=int, default=100)
@click.option('--only_nature_and_sci/--no_only_nature_and_sci', default=False)
def docs_to_keywords_df(infile, outfile, out_years, min_thresh, only_nature_and_sci):
    """
    Get dataframe of keyword frequencies over the years
    """
    # TODO: this file above should go in size folder so only one to be changed with exp

    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    df = df.drop(
        [
            "arxiv_class",
            "alternate_bibcode",
            "keyword",
            "ack",
            "aff",
            "bibstem",
            "aff_id",
            "citation_count",
        ],
        axis=1,
    )
    df['title'] = df['title'].astype(str)
    df['abstract'] = df['abstract'].astype(str)
    df['year'] = df['year'].astype(int)
    if only_nature_and_sci:
        LOG.info(f'Only looking at papers in {P_JOURNALS}')
        s0 = df.shape[0]
        df = df[df['bibcode'].apply(lambda x: x[4:8] in P_JOURNALS)]
        s1 = df.shape[0]
        LOG.info(f"Removed {s0 - s1} papers.")
    kwd_df, year_counts = flatten_to_keywords(df, min_thresh)
    LOG.info(f"Writing out all keywords to {outfile}.")
    kwd_df.to_json(outfile, orient="records", lines=True)
    LOG.info(f"Writing year counts to {out_years}.")
    year_counts.to_csv(out_years)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--out_loc", type=Path)
@click.option("--threshold", type=int)
@click.option("--score_thresh", type=float)
@click.option("--hard_limit", type=int)
def get_filtered_kwds(infile, out_loc, threshold, score_thresh, hard_limit):
    """
    Filter keywords by total frequency and rake score. Also provide hard limit.
    """
    LOG.info(f"Reading from {infile}")
    df = pd.read_json(infile, orient="records", lines=True)
    lim_kwd_df = filter_kwds(df, threshold, score_thresh, hard_limit)
    LOG.info(f"Writing dataframe with size {lim_kwd_df.shape[0]} to {out_loc}.")
    lim_kwd_df.to_json(out_loc, orient="records", lines=True)


@cli.command()
@click.option("--kwds_loc", type=Path)
@click.option("--in_years", type=Path)
@click.option("--out_norm", type=Path)
def normalize_keyword_freqs(kwds_loc, in_years, out_norm):
    """
    Normalize keyword frequencies by year totals and percent of baselines.
    """
    LOG.info(f"Reading normalized keywords years from {kwds_loc}.")
    normed_kwds_df = pd.read_json(kwds_loc, orient="records", lines=True)

    LOG.info(f"Reading year counts from {in_years}.")
    year_counts = pd.read_csv(in_years, index_col=0)

    year_counts = year_counts.sort_values("year")
    year_counts = year_counts.set_index("year")
    normed_df = normalize_by_perc(normed_kwds_df, year_counts)

    LOG.info(f"Writing normalize dataframe to {out_norm}")
    normed_df.to_json(out_norm, orient="records", lines=True)


@cli.command()
@click.option("--norm_loc", type=Path)
@click.option("--affil_loc", type=Path)
@click.option("--out_df", type=Path)
def slope_complexity(norm_loc, affil_loc, out_df):
    """
    Get various measures for keyword time series
    """
    # TODO: Variable for the path above?
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)
    overall_affil = pd.read_csv(affil_loc)['nasa_affiliation'].iloc[0]
    features = slope_count_complexity(normed_kwds_df, overall_affil)
    LOG.info(f"Writing time series features to {out_df}")
    features.to_csv(out_df)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--viz_dir", type=Path)
def plot_slope(infile, viz_dir):
    """
    Plot slope and complexity
    """
    se_df = pd.read_csv(infile, index_col=0)
    plot_slop_complex(se_df, viz_dir, x_measure="mean_change", y_measure="complexity")


@cli.command()
@click.option("--norm_loc", type=Path)
@click.option("--dtw_loc", type=Path)
def dtw(norm_loc, dtw_loc):
    """
    Compute pairwise dynamic time warp between keywords
    """
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)
    normed_kwd_years = normed_kwds_df.set_index("stem").iloc[:, 5:]
    dtw_df = dtw_kwds(normed_kwd_years)
    LOG.info(f"Outputting dynamic time warps to {dtw_loc}.")
    dtw_df.to_csv(dtw_loc)


@cli.command()
@click.option("--dtw_loc", type=Path)
@click.option("--out_elbow_plot", type=Path)
def cluster_tests(dtw_loc, out_elbow_plot):
    """
    Try various numbers of clusters for kmeans, produce plots
    """
    LOG.info(f"Reading dynamic time warp distances from {dtw_loc}.")
    dtw_df = pd.read_csv(dtw_loc, index_col=0)
    yellow_plot_kmd(dtw_df, out_elbow_plot, c_min=2, c_max=20)


@cli.command()
@click.option("--norm_loc", type=Path)
@click.option("--dtw_loc", type=Path)
@click.option("--kmeans_loc", type=Path)
@click.option("--out_man_plot", type=Path)
@click.option("--out_man_points", type=Path)
def dtw_viz(norm_loc, dtw_loc, kmeans_loc, out_man_plot, out_man_points):
    """
    Cluster keywords by dynamic time warp values and plot in tensorboard.
    """
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    kwds_df = pd.read_json(norm_loc, orient="records", lines=True)

    LOG.info(f"Reading dynamic time warp distances from {dtw_loc}.")
    dtw_df = pd.read_csv(dtw_loc, index_col=0)

    kwd_years = kwds_df.set_index("stem").iloc[:, 6:]
    # TODO: get years in different way, pulling out index by number is inflexible.
    kmeans = dtw_to_tboard(kwd_years, dtw_df, c=7)  # c taken from elbow viz
    dtw_man = dtw_to_manifold(dtw_df, out_man_plot)

    LOG.info(f"Writing kmeans model to {kmeans_loc}.")
    joblib.dump(kmeans, kmeans_loc)
    LOG.info(f"Writing manifold points to {out_man_points}.")
    joblib.dump(dtw_man, out_man_points)


if __name__ == "__main__":
    cli()
