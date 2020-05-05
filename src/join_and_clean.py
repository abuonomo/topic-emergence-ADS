import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

pandarallel.initialize()

P_JOURNALS = ["Natur", "Sci"]  # TODO: load in config


def load_records_to_dataframe(data_dir: Path, limit=None) -> pd.DataFrame:
    """
    Load records from directory to dataframe.

    Args:
        data_dir: a directory with a records from ADS. 
            Each records file is a json with keys "year", "docs", "numFound".
        limit: limit amount of records per year for testing purposes

    Returns:
        a pandas dataframe of all records from the directory
    """
    records = []
    LOG.info(f"Loading from directory: {data_dir}")
    if limit is not None:
        LOG.info(f"Limiting to {limit} records per year for testing.")
    pbar = tqdm(list(data_dir.iterdir()))
    for p in pbar:
        pbar.set_description(p.stem)
        with open(p, "r") as f0:
            r = json.load(f0)
        if limit is not None:
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
):
    df = load_records_to_dataframe(in_records_dir, limit=record_limit)
    df = df.dropna(
        subset=["abstract", "year", "nasa_afil", "title", "bibcode", "bibstem"]
    )
    allowed_db = "astronomy"
    df = df[df["database"].apply(lambda x: allowed_db in x)]
    LOG.info(f"Limited to documents in database {allowed_db}. {df.shape}")
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
    parser.add_argument("--limit", help="limit size of dataframe for testing", type=int)
    parser.add_argument(
        "--only_nature_and_sci", dest="only_nature_and_sci", action="store_true"
    )
    parser.add_argument(
        "--no_only_nature_and_sci", dest="only_nature_and_sci", action="store_false"
    )
    parser.set_defaults(only_nature_and_sci=False)
    args = parser.parse_args()
    if args.limit == 0:
        LOG.debug("Received limit of 0. Setting to None.")
        args.limit = None
    main(
        args.i,
        args.o,
        args.limit,
        only_nature_and_sci=args.only_nature_and_sci,
    )
