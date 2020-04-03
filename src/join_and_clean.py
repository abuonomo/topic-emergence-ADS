import argparse
import json
import logging
from pathlib import Path
from typing import List
from xml.sax.saxutils import unescape

import RAKE
import numpy as np
import pandas as pd
import pytextrank
import spacy
from pandarallel import pandarallel
from spacy.lang.en.stop_words import STOP_WORDS
from sqlalchemy import create_engine
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm")
TR = pytextrank.TextRank()
NLP.add_pipe(TR.PipelineComponent, name="textrank", last=True)
pandarallel.initialize()


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


def get_keywords_from_text(text: pd.Series) -> List:
    LOG.info(f"Extracting keywords from {text.shape[0]} documents.")
    tqdm.pandas()
    rake = RAKE.Rake(list(STOP_WORDS))

    def f(x):
        if type(x) == str:
            val = rake.run(x, minFrequency=1, minCharacters=3)
        else:
            val = np.nan
        return val

    rake_kwds = text.parallel_apply(f)
    return rake_kwds


def get_text_rank_kwds(text: pd.Series, batch_size=1000, n_process=1) -> List:
    LOG.info(f"Extracting keywords from {text.shape[0]} documents.")
    kwd_lists = []
    pbar = tqdm(
        NLP.pipe(text.replace(np.nan, ""), batch_size=batch_size, n_process=n_process),
        total=len(text),
    )
    for doc in pbar:
        kwds = [(p.text, p.rank) for p in doc._.phrases]
        kwd_lists.append(kwds)
        pbar.update(1)
    return kwd_lists


def write_records_to_db(df):
    db_loc = "astro2020.db"
    engine = create_engine(f"sqlite:///{db_loc}", echo=False)
    df.to_sql("ADS_records", con=engine, index=False)


def main(
    in_records_dir,
    out_records,
    record_limit=None,
    strategy="rake",
    batch_size=1000,
    n_process=1,
):
    df = load_records_to_dataframe(in_records_dir, limit=record_limit)
    df = df.dropna(subset=["abstract", "year", "nasa_afil", "title"])
    allowed_db = "astronomy"
    import ipdb; ipdb.set_trace()
    df = df[df["database"].apply(lambda x: allowed_db in x)]
    LOG.info(f"Limited to documents in database {allowed_db}. {df.shape}")
    text = df["title"] + ". " + df["abstract"]
    text = text.apply(unescape).astype(str)
    strats = ["rake", "textrank"]
    if strategy not in strats:
        raise ValueError(f"{strategy} not in {strats}.")
    if strategy == "rake":
        df["rake_kwds"] = get_keywords_from_text(text)
    elif strategy == "textrank":
        df["rake_kwds"] = get_text_rank_kwds(text, batch_size, n_process)
    LOG.info(f"Writing {len(df)} records to {out_records}.")
    df.to_json(out_records, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input raw records dir", type=Path)
    parser.add_argument("o", help="output jsonslines collected keywords", type=Path)
    parser.add_argument("--limit", help="limit size of dataframe for testing", type=int)
    parser.add_argument(
        "--strategy", help="choose either rake or textrank keyword extraction", type=str
    )
    parser.add_argument(
        "--batch_size",
        help="If textrank, choose nlp.pipe batch size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--n_process",
        help="If textrank, choose nlp.pipe number of processes",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    if args.limit == 0:
        LOG.debug("Received limit of 0. Setting to None.")
        args.limit = None
    main(args.i, args.o, args.limit, args.strategy, args.batch_size, args.n_process)
