import logging
import os
from pathlib import Path
import vaex

import joblib
import click
import pandas as pd
from scipy.io import mmwrite

from src.analyze_keyword_time_series import (
    slope_count_complexity,
    plot_slop_complex,
    plot_time,
    filter_kwds,
    dtw_to_tboard,
)
from src.create_keyword_and_syn_lists import (
    flatten_to_keywords,
    normalize,
    normalize_by_perc,
)
from src.dtw_time_analysis import dtw_kwds
from src.topic_modeling import (
    feature_and_topic_model,
    topic_model_viz,
    get_doc_len_from_file,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

MODE = os.environ["MODE"]
if MODE == "test":
    LOG.info("Testing mode")
    EXP_NAME = "kwd_analysis_perc"
    RECORDS_LOC = Path("data/rake_kwds_small.jsonl")
    MIN_THRESH = 50
    FREQ = 20
    SCORE = 1.5
    HARD = 10_000
elif MODE == "full":
    LOG.info("Full mode")
    EXP_NAME = "kwd_analysis_full_perc2"
    RECORDS_LOC = Path("data/rake_kwds.jsonl")
    MIN_THRESH = 80
    FREQ = 250
    SCORE = 1.5
    HARD = 10_000
else:
    LOG.exception("No MODE provided.")
    exit()

DATA_DIR = Path("data") / EXP_NAME
VIZ_DIR = Path("reports/viz") / EXP_NAME
MODEL_DIR = Path("models") / EXP_NAME


@click.group()
def cli():
    pass


@cli.command()
def experiment():
    """
    Run all commands in experiment.
    """
    docs_to_keywords_df()
    get_filtered_kwds()
    dtw()
    slope_complexity()
    plot_slope()
    dtw_viz()
    plot_times()


@cli.command()
def docs_to_keywords_df():
    """
    Get dataframe of keyword frequencies over the years
    """
    infile = RECORDS_LOC
    # TODO: this file above should go in size folder so only one to be changed with exp
    LOG.info(f"Reading keywords from {infile}.")
    df = pd.read_json(infile, orient="records", lines=True)
    outfile = DATA_DIR / "all_keywords.jsonl"
    out_years = DATA_DIR / "year_counts.csv"
    flatten_to_keywords(df, outfile, out_years, MIN_THRESH)


@cli.command()
def get_filtered_kwds():
    """
    Filter keywords by total frequency and rake score. Also provide hard limit.
    """
    infile = DATA_DIR / "all_keywords.jsonl"
    LOG.info(f"Reading from {infile}")
    df = pd.read_json(infile, orient="records", lines=True)
    out_loc = DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl"
    filter_kwds(df, out_loc, threshold=FREQ, score_thresh=SCORE, hard_limit=HARD)


@cli.command()
def normalize_keyword_freqs():
    """
    Normalize keyword frequencies by year totals and percent of baselines.
    """
    kwds_loc = DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl"
    in_years = DATA_DIR / "year_counts.csv"
    out_norm = DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl"

    LOG.info(f"Reading normalized keywords years from {kwds_loc}.")
    normed_kwds_df = pd.read_json(kwds_loc, orient='records', lines=True)

    LOG.info(f"Reading year counts from {in_years}.")
    year_counts = pd.read_csv(in_years, index_col=0)

    year_counts = year_counts.sort_values('year')
    year_counts = year_counts.set_index('year')
    normed_df = normalize_by_perc(normed_kwds_df, year_counts)

    LOG.info(f"Writing normalize dataframe to {out_norm}")
    normed_df.to_json(out_norm, orient='records', lines=True)


@cli.command()
def slope_complexity():
    """
    Perform linear regression and get complexity for all keywords
    """
    norm_loc = DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.csv"
    # TODO: Variable for the path above?
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_csv(norm_loc, index_col=0)
    slope_count_complexity(normed_kwds_df, DATA_DIR / "slope_complex.csv")


@cli.command()
def plot_slope():
    """
    Plot slope and complexity
    """
    infile = DATA_DIR / "slope_complex.csv"
    se_df = pd.read_csv(infile, index_col=0)
    plot_slop_complex(
        se_df, VIZ_DIR, x_measure="mean_change", y_measure="number_cwt_peaks"
    )


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
def dtw():
    """
    Compute pairwise dynamic time warp between keywords
    """
    norm_loc = DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    normed_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)
    normed_kwd_years = normed_kwds_df.set_index("stem").iloc[:, 2:]
    dtw_loc = DATA_DIR / "dynamic_time_warp_distances.csv"
    dtw_kwds(normed_kwd_years, dtw_loc)


@cli.command()
def dtw_viz():
    """
    Cluster keywords by dynamic time warp values and plot in tensorboard.
    """
    norm_loc = DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.csv"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    kwds_df = pd.read_json(norm_loc, orient="records", lines=True)
    kwd_years = kwds_df.set_index("stem").iloc[:, 2:]

    dtw_loc = DATA_DIR / "dynamic_time_warp_distances.csv"
    LOG.info(f"Reading dynamic time warp distances from {dtw_loc}.")
    dtw_df = pd.read_csv(dtw_loc, index_col=0)

    dtw_to_tboard(kwd_years, dtw_df)


@cli.command()
def make_topic_models():
    """
    Create document term matrix, topic model, and write to tensorboard
    """

    norm_loc = DATA_DIR / f"all_keywords_norm_threshold_{FREQ}_{SCORE}_{HARD}.jsonl"
    LOG.info(f"Reading normalized keywords years from {norm_loc}.")
    lim_kwds_df = pd.read_json(norm_loc, orient="records", lines=True)

    plot_loc = VIZ_DIR / "coherence.png"
    tmodels_dir = MODEL_DIR / "topic_models"
    X, mlb, mat_id_to_doc_id = feature_and_topic_model(
        lim_kwds_df, plot_loc, tmodels_dir
    )

    LOG.info("Writing matrix, multilabel binarizer, and matrix to doc id mapping.")
    MODEL_DIR.mkdir(exist_ok=True)

    mat_loc = DATA_DIR / "doc_feature_matrix.mm"
    mlb_loc = MODEL_DIR / "mlb.jbl"
    map_loc = MODEL_DIR / "mat_doc_mapping.csv"

    LOG.info(f"Writing doc feature matrix to  {mat_loc}")
    mmwrite(str(mat_loc), X)
    LOG.info(f"Writing multilabel binarizer to {mlb_loc}")
    joblib.dump(mlb, mlb_loc)
    LOG.info(f"Writing matrix to doc id mapping to {map_loc}")
    mat_id_to_doc_id.to_csv(map_loc)


@cli.command()
def visualize_topic_models():
    n_topics = 500
    tmodel_loc = MODEL_DIR / "topic_models" / f"topics_{n_topics}.jbl"
    mlb_loc = MODEL_DIR / "mlb.jbl"
    map_loc = MODEL_DIR / "mat_doc_mapping.csv"

    LOG.info(f"Counting document lengths from {RECORDS_LOC}.")
    doc_lens = get_doc_len_from_file(RECORDS_LOC)
    LOG.info(f"Loading topic model from {tmodel_loc}")
    tmodel = joblib.load(tmodel_loc)
    LOG.info(f"Loading multilabel binarizer from {mlb_loc}")
    mlb = joblib.load(mlb_loc)
    LOG.info(f"Reading matrix to doc id mapping from {map_loc}")
    mat_id_to_doc_id = pd.read_csv(map_loc, index_col=0)

    mdoc_lens = [doc_lens[i] for i in mat_id_to_doc_id["matrix_row_index"]]
    topic_model_viz(tmodel, mlb, mdoc_lens, VIZ_DIR / "topic_model_viz.html")


if __name__ == "__main__":
    cli()
