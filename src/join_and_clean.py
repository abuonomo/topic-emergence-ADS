import argparse
import yaml
import json
import logging
from pathlib import Path

import pandas as pd
from pprint import pformat
from pandarallel import pandarallel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

pandarallel.initialize()

P_JOURNALS = ["Natur", "Sci"]  # TODO: load in config


def load_records_to_dataframe(
    data_dir: Path, limit: int = None, first_year: int = None
) -> pd.DataFrame:
    """
    Load records from directory to dataframe.

    Args:
        data_dir: a directory with a records from ADS. 
            Each records file is a json with keys "year", "docs", "numFound".
        limit: limit amount of records per year for testing purposes
        first_year: earliest year to get data for

    Returns:
        a pandas dataframe of all records from the directory
    """
    records = []
    LOG.info(f"Loading from directory: {data_dir}")
    if (limit is not None) and (limit > 0):
        LOG.info(f"Limiting to {limit} records per year for testing.")
    pbar = tqdm(list(data_dir.iterdir()))
    for p in pbar:
        pbar.set_description(p.stem)
        with open(p, "r") as f0:
            r = json.load(f0)
        if first_year is not None:
            if int(r["year"]) < first_year:
                continue
        if (limit is not None) and (limit > 0):
            docs = pd.DataFrame(r["docs"][0:limit])
        else:
            docs = pd.DataFrame(r["docs"])
        docs["year"] = r["year"]
        records.append(docs)
    LOG.info("Concatenating dataframes.")
    df = pd.concat(records, sort=False)
    df = df.reset_index(drop=True)
    df["title"] = df["title"].apply(lambda x: x[0] if type(x) == list else x)
    LOG.info(f"Loaded dataframe with shape: {df.shape}")
    return df


def main(
    in_records_dir,
    out_records,
    record_limit=None,
    min_text_len=100,
    only_nature_and_sci=False,
    first_year=None,
    database_whitelist=None,
):
    df = load_records_to_dataframe(
        in_records_dir, limit=record_limit, first_year=first_year
    )
    df = df.dropna(
        subset=["abstract", "year", "nasa_afil", "title", "bibcode", "bibstem"]
    )
    if database_whitelist is not None:
        f = lambda x, y: len(set(x).intersection(set(y))) > 0
        df = df[df["database"].apply(lambda x: f(database_whitelist, x))]
        LOG.info(f"Limited to documents in databases: {database_whitelist}. {df.shape}")
    df = df[df["abstract"].apply(lambda x: len(x) >= min_text_len)]
    LOG.info(f"Limited to documents with length >= {min_text_len}. {df.shape}")
    if only_nature_and_sci:
        df = df[df["bibstem"].apply(lambda x: x[0] in P_JOURNALS)]
        LOG.info(f"Limited to documents in journals {P_JOURNALS}. {df.shape}")
    LOG.info(f"Writing {len(df)} records to {out_records}.")
    df.to_json(out_records, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input raw records dir", type=Path)
    parser.add_argument("o", help="output jsonslines collected keywords", type=Path)
    parser.add_argument("--config_loc", type=Path)
    args = parser.parse_args()

    with open(args.config_loc, 'r') as f0:
        config = yaml.safe_load(f0)
    try:
        limit = config['join_and_clean']['limit']
    except KeyError:
        limit = None
    if limit == 0:
        LOG.debug("Received limit of 0. Setting to None.")
        limit = None
    LOG.info(f"Using config:\n{pformat(config['join_and_clean'])}")
    main(
        args.i,
        args.o,
        limit,
        min_text_len=config['join_and_clean']['min_abstract_length'],
        only_nature_and_sci=False,
        first_year=None,
        database_whitelist=config['join_and_clean']['database_whitelist'],
    )
