import argparse
import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
from enstop import PLSA, EnsembleTopics
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_feature_matrix(lim_kwds_df):
    doc_to_kwd = (
        lim_kwds_df.explode("doc_id_list").groupby("doc_id_list").agg({"stem": list})
    )
    mat_id_to_doc_id = (
        doc_to_kwd.reset_index().reset_index().loc[:, ["index", "doc_id_list"]]
    )
    mat_id_to_doc_id.columns = ["matrix_row_index", "doc_id"]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(doc_to_kwd["stem"])  # inverse for docs x kwds
    return X, mlb, mat_id_to_doc_id


def topic_model_viz(model, mlb, mdoc_lens, viz_loc):
    term_freq = np.array(model.training_data_.sum(axis=0))[0]
    data = {
        "topic_term_dists": model.components_,
        "doc_topic_dists": model.embedding_,
        "vocab": mlb.classes_,
        "term_frequency": term_freq,
        "doc_lengths": mdoc_lens,
    }
    LOG.info("Preparing data for pyLDAvis")
    viz_data = pyLDAvis.prepare(**data)
    LOG.info(f"Writing visualization to {viz_loc}")
    pyLDAvis.save_html(viz_data, str(viz_loc))
    return viz_loc


def tmodel_to_tboard(X, model, doc_ids):
    writer = SummaryWriter()
    labels = model.embedding_.argmax(axis=1).tolist()
    meta = list(zip(doc_ids, [str(l) for l in labels]))
    LOG.info("Writing to tensorboard.")
    kwd_embedding = np.array(X.todense())
    writer.add_embedding(
        kwd_embedding,
        tag="keyword_embedding",
        metadata=meta,
        label_img=None,
        metadata_header=["doc_id", "topic"],
    )
    writer.close()
    LOG.info('Use "tensorboard --logdir runs" command to see visualization.')
    return model


def plot_coherence(topic_range, coherences, show=False):
    LOG.info("Plotting coherences.")
    plt.plot(topic_range, coherences)
    plt.xlabel("n_topics")
    plt.ylabel("coherence")
    plt.title("Model Coherence vs Number of Topics")
    if show:
        plt.show()
    return plt.gcf()


def feature_and_topic_model(lim_kwds_df, plot_loc, tmodels_dir, tboard=False):
    X, mlb, mat_doc_id_map = get_feature_matrix(lim_kwds_df)
    labels = mat_doc_id_map["doc_id"].tolist()
    # TODO: add train and test? But its clustering so maybe no?
    topic_range = list(range(2, 20, 5))
    coherences = []
    # TODO: instead of appending, directly write to dir of tmodels with n_topics

    LOG.info(f"Training topic models and writing to {tmodels_dir}")
    topic_pbar = tqdm(topic_range)
    for n in topic_pbar:
        topic_pbar.set_description(f"n_topics: {n}")
        # model = EnsembleTopics(n_components=n, n_jobs=12).fit(X) # TODO: make var?
        model = PLSA(n_components=n).fit(X)
        joblib.dump(model, tmodels_dir / f"topics_{n}.jbl")
        if tboard:  # will slow things down by A LOT, also does not seem to work yet
            tmodel_to_tboard(X, model, labels)
        coherences.append(model.coherence())
    fig = plot_coherence(topic_range, coherences)
    LOG.info(f"Writing plot to {plot_loc}.")
    fig.savefig(plot_loc)

    return X, mlb, mat_doc_id_map


def get_doc_length(line):
    record = json.loads(line)
    nanlen = lambda x: len(x) if x is not None else 0
    doc_len = nanlen(record["title"]) + nanlen(record["abstract"])
    return doc_len


def get_doc_len_from_file(infile):
    with open(infile, "r") as f0:
        doc_lens = [get_doc_length(line) for line in tqdm(f0)]
    return doc_lens


def main(msg, feature):
    LOG.info(f"{msg} and feature is {feature}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input txt file")
    parser.add_argument("--feature", dest="feature", action="store_true")
    parser.add_argument("--no-feature", dest="feature", action="store_false")
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.i, args.feature)
