import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import RAKE
from spacy.lang.en.stop_words import STOP_WORDS
from sqlalchemy import create_engine
from typing import List
import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


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
    df['title'] = df['title'].apply(lambda x: x[0] if type(x) == list else x)
    LOG.info(f"Loaded dataframe with shape: {df.shape}")
    return df


def get_keywords_from_text(text: pd.Series) -> List[str]:
    LOG.info(f"Extracting keywords from {text.shape[0]} documents.")
    tqdm.pandas()
    rake = RAKE.Rake(list(STOP_WORDS))

    def f(x):
        if type(x) == str:
            val = rake.run(x, minFrequency=1, minCharacters=3)
        else:
            val = np.nan
        return val

    rake_kwds = text.progress_apply(f)
    return rake_kwds


def write_records_to_db(df):
    db_loc = "astro2020.db"
    engine = create_engine(f"sqlite:///{db_loc}", echo=False)
    df.to_sql("ADS_records", con=engine, index=False)


def main(in_records_dir, out_records, record_limit=None):
    df = load_records_to_dataframe(in_records_dir, limit=record_limit)
    text = df['title'] + ' || ' + df['abstract']
    df['rake_kwds'] = get_keywords_from_text(text)
    LOG.info(f"Writing {len(df)} keywords to {out_records}.")
    df.to_json(out_records, orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input raw records dir", type=Path)
    parser.add_argument("o", help="output jsonslines collected keywords", type=Path)
    parser.add_argument("--limit", help="limit size of dataframe for testing", type=int)
    args = parser.parse_args()
    if args.limit == 0:
        LOG.debug("Received limit of 0. Setting to None.")
        args.limit = None
    main(args.i, args.o, args.limit)
