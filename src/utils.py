import argparse
import logging
import os

import h5py
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def load_topic_bibcode_h5py(infile: os.PathLike) -> pd.DataFrame:
    """
    Load h5py topic distribution with bibcodes to pandas DataFrame

    Args:
        infile: hdf5 file with bibcodes and topic distribution

    Returns:
        pandas DataFrame of topic distribution with a column of bibcodes and index which
        corresponds to the order of the records file for the topic emergence pipeline
    """
    with h5py.File(infile, "r") as f0:
        bibs = f0["bibcodes"][:]
        vals = f0["topic_distribution"][:]
        dind = f0["dist_to_doc_index"][:]
    df = pd.DataFrame(vals)
    df.insert(0, 'bibcode', bibs)
    df.index = dind
    return df


def main(infile, outfile):
    LOG.debug(f"Reading from {infile}")
    df = load_topic_bibcode_h5py(infile)

    LOG.debug(f"Writing to {outfile}")
    df.to_csv(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input h5py file of topic distributions')
    parser.add_argument('o', help='output csv of topic distributions')
    args = parser.parse_args()
    main(args.i, args.feature)
