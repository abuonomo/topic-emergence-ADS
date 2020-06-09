import json
import h5py
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, jsonify, request
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

app = Flask(__name__)

try:
    VERSION = os.environ["VERSION"]
    GIT_URL = os.environ["GIT_URL"]
except KeyError:
    VERSION = "unspecified"
    GIT_URL = "unspecified"

try:
    DATA_DIR = Path(os.environ["APP_DATA_DIR"])
except KeyError:
    DATA_DIR = Path('data')

ADS_TOKEN = os.environ['ADS_TOKEN']

app.config.update(
    SC_LOC=DATA_DIR / f"slope_complex.csv",
    N_LOC=DATA_DIR / f"all_keywords_threshold.jsonl",
    KWD_SC_LOC=DATA_DIR / f"kwd_slope_complex.csv",
    KWD_N_LOC=DATA_DIR / f"kwd_all_keywords_threshold.jsonl",
    YC_LOC=DATA_DIR / "year_counts.csv",
    KMEANS_LOC=DATA_DIR / "kmeans.jbl",
    KWD_KMEANS_LOC=DATA_DIR / "kwd_kmeans.jbl",
    MAN_LOC=DATA_DIR / "dtw_manifold_proj.jbl",
    TOPIC_DISTRIB_LOC=DATA_DIR / "topic_distribs_to_bibcodes.hdf5",
    TOPIC_YEARS_LOC=DATA_DIR / "topic_years.csv",
    VIZ_DATA_LOC=DATA_DIR / "viz_data.json",
    SC_DF=None,
    N_DF=None,
    KWD_SC_DF=None,
    KWD_N_DF=None,
    YEAR_COUNTS=None,
    KMEANS=None,
    KWD_KMEANS=None,
    TOPIC_DISTRIB_DF=None,
    TOPIC_YEARS_DF=None,
    VIZ_DATA=None,
    LOAD_COLS=[
        "stem",
        "count",
        "scaled_counts",
        "kmeans_cluster",
        "manifold_x",
        "manifold_y",
    ],
)


def get_paper_from_bibcode(bibcode):
    headers = {
        'Authorization': f"Bearer:{ADS_TOKEN}",
    }
    params = (
        ('q', f'bibcode:{bibcode}'),
        ('fl', ['title', 'abstract']),
    )
    response = requests.get('https://api.adsabs.harvard.edu/v1/search/query',
                            headers=headers, params=params)
    c = json.loads(response.content)['response']['docs'][0]
    return c


@app.before_first_request
def init():
    LOG.info(f'Reading derived time series measure from {app.config["SC_LOC"]}.')
    app.config["SC_DF"] = pd.read_csv(app.config["SC_LOC"], index_col=0)

    LOG.info(f'Reading full stem time series from {app.config["N_LOC"]}.')
    app.config["N_DF"] = pd.read_json(app.config["N_LOC"], orient="records", lines=True)

    LOG.info(f'Reading derived time series measure from {app.config["KWD_SC_LOC"]}.')
    app.config["KWD_SC_DF"] = pd.read_csv(app.config["KWD_SC_LOC"], index_col=0)

    LOG.info(f'Reading full stem time series from {app.config["KWD_N_LOC"]}.')
    app.config["KWD_N_DF"] = pd.read_json(app.config["KWD_N_LOC"], orient="records", lines=True)

    LOG.info(f"Reading kmeans model from {app.config['KMEANS_LOC']}")
    app.config["KMEANS"] = joblib.load(app.config["KMEANS_LOC"])

    LOG.info(f"Reading kmeans model from {app.config['KWD_KMEANS_LOC']}")
    app.config["KWD_KMEANS"] = joblib.load(app.config["KWD_KMEANS_LOC"])

    manifold_data = joblib.load(app.config["MAN_LOC"])
    app.config["YEAR_COUNTS"] = pd.read_csv(app.config["YC_LOC"], index_col=0)

    app.config["SC_DF"]["kmeans_cluster"] = app.config["KMEANS"].labels_
    log_count = np.log(app.config["SC_DF"]["count"])

    app.config["KWD_SC_DF"]["kmeans_cluster"] = app.config["KWD_KMEANS"].labels_
    # log_count = np.log(app.config["KWD_SC_DF"]["count"])

    scaler = MinMaxScaler(feature_range=(3, 10))
    app.config["SC_DF"]["scaled_counts"] = scaler.fit_transform(
        log_count.values.reshape(-1, 1)
    )
    app.config["SC_DF"]["manifold_x"] = manifold_data[:, 0]
    app.config["SC_DF"]["manifold_y"] = manifold_data[:, 1]

    # LOG.info(f'Reading topic distributions from {app.config["TOPIC_DISTRIB_LOC"]}')
    # app.config["TOPIC_DISTRIB_DF"] = pd.read_csv(
    #     app.config["TOPIC_DISTRIB_LOC"], index_col=0
    # ).set_index('bibcode')

    with open(app.config['VIZ_DATA_LOC'], 'r') as f0:
        app.config['VIZ_DATA'] = json.load(f0)

    LOG.info(f"Ready")


@app.route("/")
def index():
    LOG.info("Serving page.")
    return render_template("index.html", version=VERSION, git_url=GIT_URL)


@app.route("/lda", methods=["GET"])
def lda():
    LOG.info("Serving LDA viz.")
    return jsonify(app.config['VIZ_DATA'])


def load_topic_distributions(loc: os.PathLike, t: int):
    with h5py.File(loc, 'r') as f0:
        mask = f0['topic_maxes'][:] == t
        v = f0['topic_distribution'][mask, :]
        b = f0['bibcodes'][mask]

    tmp_df = pd.DataFrame(v)
    tmp_df.index = b
    tmps = tmp_df.iloc[:, t]
    df = tmps.reset_index()
    df.columns = ['bibcode', 'prob']
    return df


@app.route("/topic_bibcodes", methods=["GET", "POST"])
def topic_bibcodes():
    in_data = request.json
    topic = int(in_data['topic'])  # Frontend index starts at 1, here starts at 0
    limit = int(in_data['limit'])
    topic_df = load_topic_distributions(app.config["TOPIC_DISTRIB_LOC"], topic)
    if limit == 0:
        records = topic_df.to_dict(orient='records')
    else:
        records = topic_df.to_dict(orient='records')[0:limit]
    return jsonify(records)


@app.route("/get-scatter-data", methods=["GET", "POST"])
def get_scatter_data():
    in_data = request.json
    LOG.info(f"Getting scatter data for {in_data}.")
    cols = list(set(app.config["LOAD_COLS"] + [in_data["x"], in_data["y"]]))
    chart_data = (
        app.config["SC_DF"]
        .query(f"count >= {in_data['min_count']}")
        .query(f"score_mean >= {in_data['minRake']}")
        .loc[:, cols]
        .to_dict(orient="records")
    )
    return jsonify(chart_data)


@app.route("/doc_preview", methods=["GET", "POST"])
def get_doc_preview():
    data = request.json
    b = data['bibcode']
    d = get_paper_from_bibcode(b)
    return jsonify(d)


def _trans_time(ts, kwd, clus):
    ts["year"] = [int(v[0:4]) for v in ts.index]
    ts.columns = ["count", "year"]
    ts = ts.reset_index(drop=True)
    ts["stem"] = kwd
    ts["kmeans_cluster"] = clus

    def f(x):
        total = app.config["YEAR_COUNTS"].query(f'year == {x["year"]}').iloc[0]["count"]
        norm_val = x["count"] / total
        return norm_val

    ts["norm_count"] = ts.apply(f, axis=1)
    ts = ts.loc[:, ["stem", "kmeans_cluster", "year", "count", "norm_count"]]
    ts_recs = ts.to_dict(orient="records")
    return ts_recs


def get_time_data_inner(data, n_df, sc_df):
    LOG.info(f"Getting time data for {data}.")
    cols = [f"{y}_sum" for y in app.config['YEAR_COUNTS']['year'].sort_values()]
    ts = n_df.query(f'stem == "{data["stem"]}"').loc[:, cols]
    s = data['stem']
    kmc = sc_df.query(f'stem == "{s}"')['kmeans_cluster'].iloc[0]
    ts = ts.T
    ts_recs = _trans_time(ts, s, kmc)
    return ts_recs


@app.route("/get-time-data", methods=["GET", "POST"])
def get_time_data():
    data = request.json
    ts_recs = get_time_data_inner(data, app.config["N_DF"], app.config["SC_DF"])
    return jsonify(ts_recs)


@app.route("/get-kwd-time-data", methods=["GET", "POST"])
def get_kwd_time_data():
    data = request.json
    ts_recs = get_time_data_inner(data, app.config["KWD_N_DF"], app.config["KWD_SC_DF"])
    return jsonify(ts_recs)


@app.route("/get-all-time-data", methods=["GET", "POST"])
def get_all_time_data():
    LOG.info(f"Getting total frequencies for each year.")
    ts = pd.DataFrame(app.config["N_DF"].iloc[:, 5:].sum())
    tmp_df = app.config["YEAR_COUNTS"].copy().sort_values("year")
    ind = tmp_df["year"].apply(lambda x: f"{x}_sum")
    tmp_df["index"] = ind
    tmp_df = tmp_df.set_index("index").drop(columns=["year"])
    ts_recs = _trans_time(tmp_df, "all", 0)
    return jsonify(ts_recs)


@app.route("/get-all-options")
def get_all_options():
    LOG.info(f"Getting options for scatter axes.")
    opts = app.config["SC_DF"].columns.tolist()
    return jsonify(opts)


if __name__ == "__main__":
    app.run(debug=True)
