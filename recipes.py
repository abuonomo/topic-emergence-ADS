import logging
from pathlib import Path

import click
import pandas as pd

from src.analyze_keyword_time_series import (
    slope_count_complexity,
    plot_slop_complex,
    plot_time,
    filter_kwds,
    dtw_to_tboard,
)
from src.create_keyword_and_syn_lists import flatten_to_keywords
from src.dtw_time_analysis import dtw_kwds

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# EXP_DIR = Path("data/kwd_analysis_full_perc2")
EXP_DIR = Path("data/kwd_analysis_perc")
VIZ_DIR = Path("reports/viz")


@click.group()
def cli():
    pass


@cli.command()
def experiment():
    """
    Run all commands in experiment.
    """
    get_filtered_kwds()
    dtw()
    slope_complexity()
    plot_slope()
    dtw_viz()
    plot_times()


@cli.command()
def slope_complexity():
    """
    Perform linear regression and get complexity for all keywords
    """
    norm_loc = EXP_DIR / "all_keywords_threshold_250_1.5_10000.csv"
    # TODO: Variable for the path above?
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_csv(norm_loc, index_col=0)
    slope_count_complexity(normed_kwds_df, EXP_DIR / "slope_complex.csv")


@cli.command()
def plot_slope():
    """
    Plot slope and complexity
    """
    infile = EXP_DIR / "slope_complex.csv"
    se_df = pd.read_csv(infile, index_col=0)
    plot_slop_complex(se_df, VIZ_DIR / "slope_complex_count.html")


@cli.command()
def docs_to_keywords_df():
    """
    Get dataframe of keyword frequencies over the years
    """
    infile = Path("data/rake_kwds_small.jsonl")
    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    outfile = EXP_DIR / "all_keywords.jsonl"
    flatten_to_keywords(df, outfile)


@cli.command()
def get_filtered_kwds():
    """
    Filter keywords by total frequency and rake score. Also provide hard limit.
    """
    infile = EXP_DIR / "all_keywords.jsonl"
    LOG.info(f"Reading from {infile}")
    df = pd.read_json(infile, orient="records", lines=True)
    t = 250
    s = 1.5
    h = 10_000
    out_loc = EXP_DIR / f"all_keywords_threshold_{t}_{s}_{h}.csv"
    filter_kwds(df, out_loc, threshold=t, score_thresh=s, hard_limit=h)


@cli.command()
def plot_times():
    """
    Plot time series for individual keywords
    """
    infile = EXP_DIR / "all_keywords_limited.csv"
    LOG.info(f"Reading from {infile}")
    df = pd.read_csv(infile, index_col=0)
    inner_plot = lambda x: plot_time(x, size=(7, 7), show=True)
    lim_df = df.query("doc_id_count > 1000").set_index("stem").iloc[:, 3:]
    # Placeholder for debugging / examining the individual plots
    inner_plot(lim_df.iloc[0])


@cli.command()
def dtw():
    """
    Compute pairwise dynamic time warp between keywords
    """
    norm_loc = EXP_DIR / "all_keywords_threshold_250_1.5_10000.csv"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_csv(norm_loc, index_col=0)
    normed_kwd_years = normed_kwds_df.set_index("stem").iloc[:, 2:]
    dtw_loc = EXP_DIR / "dynamic_time_warp_distances.csv"
    dtw_kwds(normed_kwd_years, dtw_loc)


@cli.command()
def dtw_viz():
    """
    Cluster keywords by dynamic time warp values and plot in tensorboard.
    """
    norm_loc = EXP_DIR / "all_keywords_threshold_250_1.5_10000.csv"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    kwds_df = pd.read_csv(norm_loc, index_col=0)
    kwd_years = kwds_df.set_index("stem").iloc[:, 2:]

    dtw_loc = EXP_DIR / "dynamic_time_warp_distances.csv"
    LOG.info(f"Reading dynamic time warp distances from {dtw_loc}.")
    dtw_df = pd.read_csv(dtw_loc, index_col=0)

    dtw_to_tboard(kwd_years, dtw_df)


if __name__ == "__main__":
    cli()
