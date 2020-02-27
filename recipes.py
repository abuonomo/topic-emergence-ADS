import logging
from tqdm import tqdm
import os
from pathlib import Path

import click
import joblib
import pandas as pd
from scipy.io import mmwrite
from sklearn.utils import resample
import numpy as np

from src.analyze_keyword_time_series import (
    slope_count_complexity,
    plot_slop_complex,
    dtw_to_manifold,
    yellow_plot_kmd,
    plot_gm_bics,
    plot_time,
    filter_kwds,
    dtw_to_tboard,
)
import src.create_keyword_and_syn_lists as ck
from src.create_keyword_and_syn_lists import (
    flatten_to_keywords,
    normalize_by_perc,
)
from src.dtw_time_analysis import dtw_kwds
from src.topic_modeling import (
    feature_and_topic_model,
    topic_model_viz,
    get_doc_len_from_file,
)
import src.bootstrapping as bst

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

try:
    MODE = os.environ["MODE"]
except KeyError:
    MODE = "test"

if MODE == "full":
    LOG.info("Full mode")
    EXP_NAME = "kwd_analysis_full_perc2"
    RECORDS_LOC = Path("data/rake_kwds.jsonl")
    MIN_THRESH = 100
    FREQ = 250
    SCORE = 1.5
    HARD = 10_000
else:
    LOG.info("Testing mode")
    EXP_NAME = "kwd_analysis_perc"
    RECORDS_LOC = Path("data/rake_kwds_small.jsonl")
    MIN_THRESH = 500
    FREQ = 20
    SCORE = 1.5
    HARD = 10_000

DATA_DIR = Path("data") / EXP_NAME
VIZ_DIR = Path("reports/viz") / EXP_NAME
MODEL_DIR = Path("models") / EXP_NAME


@click.group()
def cli():
    pass


@cli.command()
@click.option("--infile", type=Path, default=RECORDS_LOC)
@click.option("--outfile", type=Path, default=DATA_DIR / "all_keywords.jsonl")
@click.option("--out_years", type=Path, default=DATA_DIR / "year_counts.csv")
def prepare_bootstrap(infile, outfile, out_years, min_thresh=100):
    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    kwd_df_resamples, year_counts = bst.get_resamples(df, min_thresh=500)
    # TODO: feed bootstraps into full pipeline to create all tsfresh metrics, this time with a distribution. How to visualize with distribution?
    # TODO: get time estimate on how long that would take, knowing how fast one is shoudl be a good approximation.
    for kwd_df in kwd_df_resamples:
        kwd_df = kwd_df.reset_index()
        LOG.info(f"Writing out all keywords to {outfile}.")
        kwd_df.to_json(outfile, orient="records", lines=True)
        LOG.info(f"Writing year counts to {out_years}.")
        year_counts.to_csv(out_years)



@cli.command()
@click.option("--infile", type=Path, default=RECORDS_LOC)
@click.option("--outfile", type=Path, default=DATA_DIR / "all_keywords.jsonl")
@click.option("--out_years", type=Path, default=DATA_DIR / "year_counts.csv")
def docs_to_keywords_df(infile, outfile, out_years):
    """
    Get dataframe of keyword frequencies over the years
    """
    # TODO: this file above should go in size folder so only one to be changed with exp

    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    kwd_df, year_counts = flatten_to_keywords(df, MIN_THRESH)
    LOG.info(f"Writing out all keywords to {outfile}.")
    kwd_df.to_json(outfile, orient="records", lines=True)
    LOG.info(f"Writing year counts to {out_years}.")
    year_counts.to_csv(out_years)


@cli.command()
@click.option("--infile", type=Path, default=DATA_DIR / "all_keywords.jsonl")
@click.option(
    "--out_loc",
    type=Path,
    default=DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
@click.option("--threshold", type=int, default=FREQ)
@click.option("--score_thresh", type=float, default=SCORE)
@click.option("--hard_limit", type=int, default=HARD)
def get_filtered_kwds(infile, out_loc, threshold, score_thresh, hard_limit):
    """
    Filter keywords by total frequency and rake score. Also provide hard limit.
    """
    LOG.info(f"Reading from {infile}")
    df = pd.read_json(infile, orient="records", lines=True)
    filter_kwds(df, out_loc, threshold, score_thresh, hard_limit)


@cli.command()
@click.option(
    "--kwds_loc",
    type=Path,
    default=DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
@click.option(
    "--in_years", type=Path, default=DATA_DIR / "year_counts.csv",
)
@click.option(
    "--out_norm",
    type=Path,
    default=DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
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
@click.option(
    "--norm_loc",
    type=Path,
    default=DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
@click.option(
    "--out_df", type=Path, default=DATA_DIR / "slope_complex.csv",
)
def slope_complexity(norm_loc, out_df):
    """
    Get various measures for keyword time series
    """
    # TODO: Variable for the path above?
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)
    slope_count_complexity(normed_kwds_df, out_df)


@cli.command()
@click.option("--infile", type=Path, default=DATA_DIR / "slope_complex.csv")
@click.option("--viz_dir", type=Path, default=VIZ_DIR)
def plot_slope(infile, viz_dir):
    """
    Plot slope and complexity
    """
    se_df = pd.read_csv(infile, index_col=0)
    plot_slop_complex(se_df, viz_dir, x_measure="mean_change", y_measure="complexity")


@cli.command()
def plot_times():
    """
    Plot time series for individual keywords
    """
    infile = DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl"
    LOG.info(f"Reading from {infile}")
    df = pd.read_json(infile, orient="records", lines=True)
    inner_plot = lambda x: plot_time(x, size=(7, 7), show=True)
    lim_df = df.set_index("stem").iloc[:, 5:]
    # Placeholder for debugging / examining the individual plots
    inner_plot(lim_df.iloc[0])


@cli.command()
@click.option(
    "--norm_loc",
    type=Path,
    default=DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
@click.option(
    "--dtw_loc", type=Path, default=DATA_DIR / "dynamic_time_warp_distances.csv"
)
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
@click.option(
    "--dtw_loc", type=Path, default=DATA_DIR / "dynamic_time_warp_distances.csv"
)
@click.option("--out_elbow_plot", type=Path, default=VIZ_DIR / "elbow.png")
def cluster_tests(dtw_loc, out_elbow_plot):
    """
    Try various numbers of clusters for kmeans, produce plots
    """
    LOG.info(f"Reading dynamic time warp distances from {dtw_loc}.")
    dtw_df = pd.read_csv(dtw_loc, index_col=0)
    yellow_plot_kmd(dtw_df, out_elbow_plot, c_min=2, c_max=20)


@cli.command()
@click.option(
    "--norm_loc",
    type=Path,
    default=DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
@click.option(
    "--dtw_loc", type=Path, default=DATA_DIR / "dynamic_time_warp_distances.csv",
)
@click.option(
    "--kmeans_loc", type=Path, default=MODEL_DIR / "kmeans.jbl",
)
@click.option(
    "--out_man_plot", type=Path, default=VIZ_DIR / "manifold.png",
)
@click.option(
    "--out_man_points", type=Path, default=MODEL_DIR / "dtw_manifold_proj.jbl",
)
def dtw_viz(norm_loc, dtw_loc, kmeans_loc, out_man_plot, out_man_points):
    """
    Cluster keywords by dynamic time warp values and plot in tensorboard.
    """
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    kwds_df = pd.read_json(norm_loc, orient="records", lines=True)

    LOG.info(f"Reading dynamic time warp distances from {dtw_loc}.")
    dtw_df = pd.read_csv(dtw_loc, index_col=0)

    kwd_years = kwds_df.set_index("stem").iloc[:, 5:]
    kmeans = dtw_to_tboard(kwd_years, dtw_df, c=7)  # c taken from elbow viz
    dtw_man = dtw_to_manifold(dtw_df, out_man_plot)

    LOG.info(f"Writing kmeans model to {kmeans_loc}.")
    joblib.dump(kmeans, kmeans_loc)
    LOG.info(f"Writing manifold points to {out_man_points}.")
    joblib.dump(dtw_man, out_man_points)


@cli.command()
@click.option(
    "--norm_loc",
    type=Path,
    default=DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
)
@click.option(
    "--plot_loc", type=Path, default=VIZ_DIR / "coherence.png",
)
@click.option(
    "--mat_loc", type=Path, default=DATA_DIR / "doc_feature_matrix.mm",
)
@click.option(
    "--mlb_loc", type=Path, default=MODEL_DIR / "mlb.jbl",
)
@click.option(
    "--map_loc", type=Path, default=MODEL_DIR / "mat_doc_mapping.csv",
)
@click.option(
    "--tmodels_dir", type=Path, default=MODEL_DIR / "topic_models",
)
def make_topic_models(norm_loc, plot_loc, mat_loc, mlb_loc, map_loc, tmodels_dir):
    """
    Create document term matrix, topic model, and write to tensorboard
    """
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    lim_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)

    X, mlb, mat_id_to_doc_id = feature_and_topic_model(
        lim_kwds_df, plot_loc, tmodels_dir
    )

    LOG.info("Writing matrix, multilabel binarizer, and matrix to doc id mapping.")
    MODEL_DIR.mkdir(exist_ok=True)

    LOG.info(f"Writing doc feature matrix to  {mat_loc}")
    mmwrite(str(mat_loc), X)
    LOG.info(f"Writing multilabel binarizer to {mlb_loc}")
    joblib.dump(mlb, mlb_loc)
    LOG.info(f"Writing matrix to doc id mapping to {map_loc}")
    mat_id_to_doc_id.to_csv(map_loc)


@cli.command()
@click.option("--tmodel_dir", type=Path, default=MODEL_DIR / "topic_models")
@click.option("--n", type=int, default=7)
@click.option("--mlb_loc", type=Path, default=MODEL_DIR / "mlb.jbl")
@click.option("--map_loc", type=Path, default=MODEL_DIR / "mat_doc_mapping.csv")
@click.option("--tmodel_viz_loc", type=Path, default=VIZ_DIR / "topic_model_viz.html")
def visualize_topic_models(tmodel_dir, n, mlb_loc, map_loc, tmodel_viz_loc):
    tmodel_loc = tmodel_dir / f"topics_{n}.jbl"
    LOG.info(f"Counting document lengths from {RECORDS_LOC}.")
    doc_lens = get_doc_len_from_file(RECORDS_LOC)
    LOG.info(f"Loading topic model from {tmodel_loc}")
    tmodel = joblib.load(tmodel_loc)
    LOG.info(f"Loading multilabel binarizer from {mlb_loc}")
    mlb = joblib.load(mlb_loc)
    LOG.info(f"Reading matrix to doc id mapping from {map_loc}")
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)

    mdoc_lens = [doc_lens[i] for i in mat_id_to_doc_id["matrix_row_index"]]
    topic_model_viz(tmodel, mlb, mdoc_lens, tmodel_viz_loc)


if __name__ == "__main__":
    cli()
