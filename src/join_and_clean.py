import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import RAKE.RAKE as rr
from spacy.lang.en.stop_words import STOP_WORDS
from sqlalchemy import create_engine
from typing import List
import operator

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def generate_candidate_keyword_scores(phrase_list, word_score, minFrequency):
    keyword_candidates = {}
    for phrase in tqdm(phrase_list):
        if phrase_list.count(phrase) >= minFrequency:
            keyword_candidates.setdefault(phrase, 0)
            word_list = rr.separate_words(phrase)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            keyword_candidates[phrase] = candidate_score
    return keyword_candidates


class DsRake(rr.Rake):
    def __init__(self, stop_words):
        super().__init__(stop_words)

    def dsrun(self, text, minCharacters=1, maxWords=5, minFrequency=1):
        sentence_list = tqdm(rr.split_sentences(text))
        LOG.info("Getting phrase list.")
        phrase_list = rr.generate_candidate_keywords(
            sentence_list, self._Rake__stop_words_pattern, minCharacters, maxWords
        )
        LOG.info("Calculating word scores.")
        word_scores = rr.calculate_word_scores(tqdm(phrase_list))
        LOG.info("Generating candidate keyword scores.")
        keyword_candidates = generate_candidate_keyword_scores(
            phrase_list, word_scores, minFrequency
        )
        sorted_keywords = sorted(
            keyword_candidates.items(), key=operator.itemgetter(1), reverse=True
        )
        return sorted_keywords


def load_records_to_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Load records from directory to dataframe.

    Args:
        data_dir: a directory with a records from ADS. 
        Each records file is a json with keys "year", "docs", "numFound".

    Returns:
        a pandas dataframe of all records from the directory
    """
    records = []
    LOG.info(f"Loading from directory: {data_dir}")
    pbar = tqdm(list(data_dir.iterdir()))
    for p in pbar:
        pbar.set_description(p.stem)
        with open(p, "r") as f0:
            r = json.load(f0)
        docs = pd.DataFrame(r["docs"])
        docs["year"] = r["year"]
        records.append(docs)
    LOG.info("Concatenating dataframes.")
    df = pd.concat(records, sort=False)
    df = df.reset_index(drop=True)
    LOG.info(f"Loaded dataframe with shape: {df.shape}")
    return df


def get_keywords_from_text(text: pd.Series) -> List[str]:
    LOG.info(f"Extracting keywords from {text.shape[0]} documents.")
    tqdm.pandas()
    rake = rr.Rake(list(STOP_WORDS))
    rake_kwds = text.progress_apply(lambda x: rake.run(x, minFrequency=1, minCharacters=3))
    # TODO: determine parameters for this function
    # Maybe make heuristic which depends upon the number of docs in corpus
    return rake_kwds


def write_records_to_db(df):
    db_loc = "astro2020.db"
    engine = create_engine(f"sqlite:///{db_loc}", echo=False)
    df.to_sql("ADS_records", con=engine, index=False)


def main(in_records_dir, out_keywords, record_limit=None):
    df = load_records_to_dataframe(in_records_dir)
    if record_limit is not None:
        LOG.info(f"Limit to {record_limit} records for testing.")
        df = df.iloc[0:record_limit]
    title = df["title"].apply(lambda x: x[0] if type(x) == list else "")
    abstract = df["abstract"].astype(str)
    text = abstract + title
    rake_kwds = get_keywords_from_text(text)
    LOG.info(f"Writing {len(rake_kwds)} keywords to {out_keywords}.")
    with open(out_keywords, "w") as f0:
        for r in tqdm(rake_kwds):
            f0.write(json.dumps(r))
            f0.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input raw records dir", type=Path)
    parser.add_argument("o", help="output jsonslines collected keywords", type=Path)
    parser.add_argument("--limit", help="limit size of dataframe for testing", type=int)
    args = parser.parse_args()
    main(args.i, args.o, args.limit)
