import json
import logging
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

DATA_DIR = Path("data")
FREQ = 250  # Test FREQ = 20
SCORE = 1.5
HARD = 10000

app.config.update(
    SC_LOC=DATA_DIR / f"slope_complex.csv",
    N_LOC=DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
    KMEANS_LOC=DATA_DIR / "kmeans.jbl",
    MAN_LOC=DATA_DIR / "dtw_manifold_proj.jbl",
    SC_DF=None,
    N_DF=None,
    KMEANS=None,
)


@app.before_first_request
def init():
    LOG.info(f'Reading derived time series measure from {app.config["SC_LOC"]}.')
    app.config["SC_DF"] = pd.read_csv(app.config["SC_LOC"], index_col=0)
    LOG.info(f'Reading full keyword time series from {app.config["N_LOC"]}.')
    app.config["N_DF"] = pd.read_json(app.config["N_LOC"], orient="records", lines=True)
    LOG.info(f"Reading kmeans model from {app.config['KMEANS_LOC']}")
    app.config['KMEANS'] = joblib.load(app.config['KMEANS_LOC'])
    manifold_data = joblib.load(app.config["MAN_LOC"])
    app.config['SC_DF']['kmeans_cluster'] = app.config['KMEANS'].labels_
    log_count = np.log(app.config['SC_DF']['count'])

    scaler = MinMaxScaler(feature_range=(3, 10))
    app.config['SC_DF']['scaled_counts'] = scaler.fit_transform(log_count.values.reshape(-1, 1))
    app.config["SC_DF"]['manifold_x'] = manifold_data[:, 0]
    app.config["SC_DF"]['manifold_y'] = manifold_data[:, 1]


@app.route("/")
def index():
    chart_data = app.config["SC_DF"].to_dict(orient="records")
    chart_data = json.dumps(chart_data, indent=2)
    return render_template("index.html", data=chart_data)


@app.route("/get-scatter-data", methods=['GET'])
def get_scatter_data():
    chart_data = app.config["SC_DF"].to_dict(orient="records")
    return jsonify(chart_data)


@app.route("/get-time-data", methods=["GET", "POST"])
def get_time_data():
    data = request.json
    ts = app.config['N_DF'].query(f'stem == \"{data["keyword"]}\"').iloc[:, 5:]
    ts = ts.T
    ts['year'] = [int(v[0:4]) for v in ts.index]
    ts.columns = ['count', 'year']
    ts = ts.reset_index(drop=True)
    ts['keyword'] = data['keyword']
    ts['kmeans_cluster'] = data['kmeans_cluster']
    ts = ts.loc[:, ['keyword', 'kmeans_cluster', 'year', 'count']]
    ts_recs = ts.to_dict(orient="records")
    return jsonify(ts_recs)


if __name__ == "__main__":
    app.run(debug=True)
