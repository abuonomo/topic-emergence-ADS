import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

DATA_DIR = Path(os.environ["APP_DATA_DIR"])

app.config.update(
    SC_LOC=DATA_DIR / f"slope_complex.csv",
    N_LOC=DATA_DIR / f"all_keywords_threshold.jsonl",
    YC_LOC=DATA_DIR / "year_counts.csv",
    KMEANS_LOC=DATA_DIR / "kmeans.jbl",
    MAN_LOC=DATA_DIR / "dtw_manifold_proj.jbl",
    TOPIC_DISTRIB_LOC=DATA_DIR / "topic_distribs_to_bibcodes.csv",
    TOPIC_YEARS_LOC=DATA_DIR / "topic_years.csv",
    SC_DF=None,
    N_DF=None,
    YEAR_COUNTS=None,
    KMEANS=None,
    TOPIC_DISTRIB_DF=None,
    TOPIC_YEARS_DF=None,
    LOAD_COLS=[
        "stem",
        "count",
        "scaled_counts",
        "kmeans_cluster",
        "manifold_x",
        "manifold_y",
    ],
)


@app.before_first_request
def init():
    LOG.info(f'Reading derived time series measure from {app.config["SC_LOC"]}.')
    app.config["SC_DF"] = pd.read_csv(app.config["SC_LOC"], index_col=0)

    LOG.info(f'Reading full stem time series from {app.config["N_LOC"]}.')
    app.config["N_DF"] = pd.read_json(app.config["N_LOC"], orient="records", lines=True)

    LOG.info(f"Reading kmeans model from {app.config['KMEANS_LOC']}")
    app.config["KMEANS"] = joblib.load(app.config["KMEANS_LOC"])

    manifold_data = joblib.load(app.config["MAN_LOC"])
    app.config["YEAR_COUNTS"] = pd.read_csv(app.config["YC_LOC"], index_col=0)

    app.config["SC_DF"]["kmeans_cluster"] = app.config["KMEANS"].labels_
    log_count = np.log(app.config["SC_DF"]["count"])

    scaler = MinMaxScaler(feature_range=(3, 10))
    app.config["SC_DF"]["scaled_counts"] = scaler.fit_transform(
        log_count.values.reshape(-1, 1)
    )
    app.config["SC_DF"]["manifold_x"] = manifold_data[:, 0]
    app.config["SC_DF"]["manifold_y"] = manifold_data[:, 1]

    # LOG.info(f'Reading topic distributions from {app.config["TOPIC_DISTRIB_LOC"]}')
    # app.config["TOPIC_DISTRIB_DF"] = pd.read_csv(
    #     app.config["TOPIC_DISTRIB_LOC"], index_col=0
    # )

    # app.config["TOPIC_YEARS_DF"] = pd.read_csv(
    #     app.config["TOPIC_YEARS_LOC"], index_col=0
    # )
    LOG.info(f"Ready")


@app.route("/")
def index():
    LOG.info("Serving page.")
    return render_template("index.html", version=VERSION, git_url=GIT_URL)


@app.route("/lda")
def lda():
    LOG.info("Serving LDA viz.")
    return render_template("lda.html", version=VERSION, git_url=GIT_URL)


@app.route("/topic_bibcodes", methods=["GET", "POST"])
def topic_bibcodes():
    import ipdb; ipdb.set_trace()
    in_data = request.json
    topic = in_data['topic'] - 1  # Frontend index starts at 1, here starts at 0
    # norm_counts = app.config['TOPIC_YEARS_DF'] / app.config['TOPIC_YEARS_DF'].sum()
    # ts = app.config['TOPIC_YEARS_DF'].loc[topic, :]
    # ts_norm = norm_counts.loc[topic, :]
    # df = pd.DataFrame({'count': ts, 'norm_count': ts_norm})
    # df['stem'] = str(topic)
    # df['kmeans_cluster'] = 1
    # df = df.reset_index().rename(columns={'index': 'year'})
    # df['year'] = df['year'].astype(int)
    # records = df.to_dict(orient='records')
    # return jsonify(records)


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
    LOG.info(f"Getting time data for {data}.")
    ts = app.config["N_DF"].query(f'stem == "{data["stem"]}"').iloc[:, 6:]
    s = data['stem']
    kmc = app.config['SC_DF'].query(f'stem == "{s}"')['kmeans_cluster'].iloc[0]
    ts = ts.T
    ts_recs = _trans_time(ts, s, kmc)
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
