import json
import h5py
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import requests
from flask import Flask, render_template, jsonify, request
from sklearn.preprocessing import MinMaxScaler

from model import VizPrepper

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

app = Flask(__name__)

try:
    VERSION = os.environ["VERSION"]
except KeyError:
    VERSION = "unspecified"

try:
    GIT_URL = os.environ["GIT_URL"]
except KeyError:
    GIT_URL = "unspecified"

LOG.info(f"Deploying Version: {VERSION}")

try:
    DATA_DIR = Path(os.environ["APP_DATA_DIR"])
except KeyError:
    DATA_DIR = Path('data')

ADS_TOKEN = os.environ['ADS_TOKEN']

app.config.update(
    PARAM_CONFIG_LOC=DATA_DIR / "config.yaml",
    VIZ_DATA_LOC=DATA_DIR / "viz_data.hdf5",
    VP=None,
    SC_LOC=DATA_DIR / f"time_series_characteristics.csv",
    N_LOC=DATA_DIR / f"topic_years.csv",
    # KWD_SC_LOC=DATA_DIR / f"kwd_slope_complex.csv",
    # KWD_N_LOC=DATA_DIR / f"kwd_all_keywords_threshold.jsonl",
    YC_LOC=DATA_DIR / "year_counts.csv",
    # KMEANS_LOC=DATA_DIR / "kmeans.jbl",
    # KWD_KMEANS_LOC=DATA_DIR / "kwd_kmeans.jbl",
    # MAN_LOC=DATA_DIR / "dtw_manifold_proj.jbl",
    # TOPIC_DISTRIB_LOC=DATA_DIR / "topic_distribs_to_bibcodes.hdf5",
    TOPIC_YEARS_LOC=DATA_DIR / "topic_years.csv",
    PYLDAVIS_DATA_LOC=DATA_DIR / "pyLDAvis_data.json",
    PYLDAVIS_DATA=None,
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

def load_kwd_ts_df(viz_data_loc):
    with h5py.File(viz_data_loc, "r") as f0:
        keywords = f0["keywords"][:]
        kwd_ts_values = f0["keyword_ts_values"][:]
    kwd_ts_df = pd.DataFrame(kwd_ts_values)
    kwd_ts_df.index = keywords
    return kwd_ts_df

@app.before_first_request
def init():
    # with open(app.config['PARAM_CONFIG_LOC'], 'r') as f0:
    #     config = yaml.load(f0)

    # app.config['VP'] = VizPrepper()
    # app.config['VP'].read_hdf(app.config['VIZ_DATA_LOC'])
    # app.config['VP'].read_pyldavis_data(app.config['PYLDAVIS_DATA_LOC'])
    # app.config['PYLDAVIS_DATA'] = app.config['VP'].pyLDAvis_data
    with open(app.config['PYLDAVIS_DATA_LOC'], 'r') as f0:
        app.config['PYLDAVIS_DATA'] = json.load(f0)

    # ts_df, features_df = app.config['VP'].get_time_characteristics(**config['app'])
    # kmeans, dtw_man = app.config['VP'].get_dynamic_time_warp_clusters(ts_df)
    # app.config['SC_DF'] = features_df
    # app.config['SC_DF']['kmeans_cluster'] = kmeans.labels_

    LOG.info(f'Reading derived time series measure from {app.config["SC_LOC"]}.')
    app.config["SC_DF"] = pd.read_csv(app.config["SC_LOC"], index_col=0)

    # app.config["N_DF"] = pd.read_json(app.config["N_LOC"], orient="records", lines=True)
    app.config["N_DF"] = pd.read_csv(app.config["N_LOC"], index_col=0)

    # LOG.info(f'Reading derived time series measure from {app.config["KWD_SC_LOC"]}.')
    # app.config["KWD_SC_DF"] = pd.read_csv(app.config["KWD_SC_LOC"], index_col=0)

    # LOG.info(f'Reading full stem time series from {app.config["KWD_N_LOC"]}.')
    # app.config["KWD_N_DF"] = pd.read_json(app.config["KWD_N_LOC"], orient="records", lines=True)
    app.config['KWD_N_DF'] = load_kwd_ts_df(app.config['VIZ_DATA_LOC'])
    # app.config['KWD_N_DF'] = app.config['VP'].kwd_ts_df

    # LOG.info(f"Reading kmeans model from {app.config['KMEANS_LOC']}")
    # app.config["KMEANS"] = joblib.load(app.config["KMEANS_LOC"])

    # LOG.info(f"Reading kmeans model from {app.config['KWD_KMEANS_LOC']}")
    # app.config["KWD_KMEANS"] = joblib.load(app.config["KWD_KMEANS_LOC"])

    # manifold_data = joblib.load(app.config["MAN_LOC"])
    app.config["YEAR_COUNTS"] = pd.read_csv(app.config["YC_LOC"], index_col=0)

    # app.config["SC_DF"]["kmeans_cluster"] = app.config["KMEANS"].labels_
    log_count = np.log(app.config["SC_DF"]["count"])
    log_count = log_count.replace([np.inf, -np.inf], 0)

    # app.config["KWD_SC_DF"]["kmeans_cluster"] = app.config["KWD_KMEANS"].labels_
    # app.config["KWD_SC_DF"]["kmeans_cluster"] = app.config["KWD_KMEANS"].labels_
    # log_count = np.log(app.config["KWD_SC_DF"]["count"])

    scaler = MinMaxScaler(feature_range=(3, 10))
    app.config["SC_DF"]["scaled_counts"] = scaler.fit_transform(
        log_count.values.reshape(-1, 1)
    )
    # app.config["SC_DF"]["manifold_x"] = dtw_man[:, 0]
    # app.config["SC_DF"]["manifold_y"] = dtw_man[:, 1]

    # with open(app.config['PYLDAVIS_DATA_LOC'], 'r') as f0:
    #     app.config['PYLADVIS_DATA'] = json.load(f0)

    LOG.info(f"Ready")


@app.route("/")
def index():
    LOG.info("Serving page.")
    return render_template("index.html", version=VERSION, git_url=GIT_URL)


@app.route("/lda", methods=["GET"])
def lda():
    LOG.info("Serving LDA viz.")
    return jsonify(app.config['PYLDAVIS_DATA'])


def load_topic_distributions(loc: os.PathLike, t: int):
    with h5py.File(loc, 'r') as f0:
        mask = f0['topic_maxes'][:] == t
        v = f0['embedding'][mask, :]
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
    topic_df = load_topic_distributions(app.config["VIZ_DATA_LOC"], topic)
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
    not_in_cols = [c for c in cols if c not in app.config['SC_DF'].columns]
    if len(not_in_cols) > 0:
        raise ValueError("All LOAD_COLS must be in SC_DF")
    chart_data = (
        app.config["SC_DF"]
        .query(f"count > 0")
        .query(f"count >= {in_data['min_count']}")
        .query(f"coherence_score >= {in_data['minRake']}")
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


@app.route("/get-time-data", methods=["GET", "POST"])
def get_time_data():
    data = request.json
    df = app.config['N_DF'].loc[int(data['stem'])].reset_index()
    df.columns = ['year', 'count']
    df['kmeans_cluster'] = app.config['SC_DF']['kmeans_cluster'].loc[int(data['stem'])]
    records = df.to_dict(orient='records')
    return jsonify(records)


@app.route("/get-kwd-time-data", methods=["GET", "POST"])
def get_kwd_time_data():
    data = request.json
    df = app.config['KWD_N_DF'].loc[data['stem']].reset_index()
    df.columns = ['year', 'count']
    records = df.to_dict(orient='records')
    return jsonify(records)


@app.route("/get-all-time-data", methods=["GET", "POST"])
def get_all_time_data():
    LOG.info(f"Getting total frequencies for each year.")
    df = app.config["N_DF"].sum().reset_index()
    df.columns = ['year', 'count']
    records = df.to_dict(orient='records')
    return jsonify(records)


@app.route("/get-all-options")
def get_all_options():
    LOG.info(f"Getting options for scatter axes.")
    opts = app.config["SC_DF"].columns.tolist()
    return jsonify(opts)


if __name__ == "__main__":
    app.run(debug=True)
