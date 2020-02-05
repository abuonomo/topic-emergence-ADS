import json
from pathlib import Path

from flask import Flask, render_template, jsonify, request
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

app = Flask(__name__)

DATA_DIR = Path("data")
FREQ = 20
SCORE = 1.5
HARD = 10000

app.config.update(
    SC_LOC=DATA_DIR / f"slope_complex.csv",
    N_LOC=DATA_DIR / f"all_keywords_threshold_{FREQ}_{SCORE}_{HARD}.jsonl",
    SC_DF=None,
    N_DF=None,
)


@app.before_first_request
def init():
    LOG.info(f'Reading derived time series measure from {app.config["SC_LOC"]}.')
    app.config["SC_DF"] = pd.read_csv(app.config["SC_LOC"], index_col=0)
    LOG.info(f'Reading full keyword time series from {app.config["N_LOC"]}.')
    app.config["N_DF"] = pd.read_json(app.config["N_LOC"], orient="records", lines=True)


@app.route("/")
def index():
    chart_data = app.config["SC_DF"].to_dict(orient="records")
    chart_data = json.dumps(chart_data, indent=2)
    data = {"chart_data": chart_data}
    return render_template("index.html", data=data)


@app.route("/get-data", methods=["GET", "POST"])
def get_data():
    data = request.json
    ts = app.config['N_DF'].query(f'stem == \"{data["keyword"]}\"').iloc[:, 5:]
    ts = ts.T
    ts['year'] = [int(v[0:4]) for v in ts.index]
    ts.columns = ['count', 'year']
    ts = ts.reset_index(drop=True)
    ts['keyword'] = data['keyword']
    ts = ts.loc[:, ['keyword', 'year', 'count']]
    ts_recs = ts.to_dict(orient="records")
    return jsonify(ts_recs)


if __name__ == "__main__":
    app.run(debug=True)
