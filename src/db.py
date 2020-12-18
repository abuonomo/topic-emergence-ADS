import json
import logging
import sqlite3
from contextlib import closing
from html import unescape
from html.parser import HTMLParser
from math import ceil
from pathlib import Path
from pprint import pformat
from typing import Dict, Union

import click
import dask
import numpy as np
import spacy
import scispacy
import yaml
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.matutils import Sparse2Corpus
from scipy.sparse import coo_matrix
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Float
from sqlalchemy import create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from textacy.ke import textrank
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

BASE = declarative_base()


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return "".join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class PaperKeywords(BASE):
    __tablename__ = "paper_keywords"

    paper_id = Column(
        Integer, ForeignKey("papers.id"), primary_key=True, nullable=False
    )
    keyword_id = Column(
        Integer, ForeignKey("keywords.id"), primary_key=True, nullable=False
    )
    score = Column(Float)
    raw_keyword = Column(String, index=True)
    # TODO: redundant in current implementation, same as keyword.keyword
    count = Column(Integer)
    paper = relationship("Paper", back_populates="keywords")
    keyword = relationship("Keyword", back_populates="papers")

    def __repr__(self):
        return f'<PaperKeywords(paper_id="{self.paper_id}", keyword.keyword="{self.keyword.keyword}")>'

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_sa_instance_state", "keyword"]
        }


def clean_text(text):
    text = unescape(text)
    text = strip_tags(text)
    return text


class Paper(BASE):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bibcode = Column(String(19), index=True)
    title = Column(String)
    abstract = Column(String)
    citation_count = Column(Integer)
    year = Column(Integer, index=True)
    nasa_affiliation = Column(Boolean)
    lemma_text = Column(String)
    keywords = relationship("PaperKeywords", back_populates="paper")

    def __repr__(self):
        return f'<Paper(bibcode="{self.bibcode}", title="{self.title}")>'

    def get_feature_text(self):
        text = f"{self.title}. {self.abstract}"
        text = clean_text(text)
        return text

    def to_dict(self):
        d = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_sa_instance_state", "keywords"]
        }
        d["keyword_ids"] = [k.keyword_id for k in self.keywords]
        return d


class Keyword(BASE):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String, nullable=False, unique=True, index=True)
    papers = relationship("PaperKeywords", back_populates="keyword")

    def __init__(self, keyword):
        self.keyword = keyword

    def __repr__(self):
        return f'<Keyword(keyword="{self.keyword}")>'

    def get_years(self, session):
        years = (
            session.query(Paper.year, func.count(Paper.year))
            .join(PaperKeywords)
            .join(Keyword)
            .filter(Keyword.keyword == self.keyword)
            .group_by(Paper.year)
            .all()
        )
        return years


def is_nu_like(s):
    """
    Check if the given string looks like a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_keyword_from_doc(doc):
    """
    Extract keywords from a single spacy doc using SingleRank

    Args:
        doc: Spacy doc from which to extract keywords

    Returns:
        text: The lemmatized and lowercase text for the doc
        kwd_counts: The SingleRank keywords with their scores and counts.
    """
    # SingleRank parameters
    kwds = textrank(
        doc,
        normalize="lemma",
        topn=999_999,  # This could cause issues with a huge abstract
        window_size=10,
        edge_weighting="count",
        position_bias=False,
    )
    # Remove keywords which are 1 character long or are numbers
    kwds = [(k.lower(), v) for k, v in kwds if (not is_nu_like(k)) and (len(k) > 1)]
    text = " ".join([t.lemma_.lower() for t in doc])
    kwd_counts = [(k, v, text.count(k)) for k, v in kwds]
    # TODO: "Black hole nos with xnosnosx" would count "nos" 3 times. Do we want this?
    # If make match " nos ", need to keep in mind problems at beginning and end
    # of a sentence, and around punctuation, parentheses, etc.
    return text, kwd_counts


class PaperKeywordExtractor:
    def __init__(self, nlp):
        """
        Object for extracting keywords from papers.

        Args:
            nlp: spacy model to use for keyword extraction
        """
        self.nlp = nlp

    def extract_all_keywords(
        self, session, batch_size=100, n_process=-1,
    ):
        """
        Extract keywords from all papers using SingleRank. Then insert these keywords
            into the database.

        Args:
            session: sqlalchemy session connected to database
            batch_size: batch_size for spacy model pipe
            n_process: number of processes for spacy model pipe
        """
        paper_query = session.query(Paper)
        nu_papers = paper_query.count()
        LOG.info(f"Extracting keywords from {nu_papers} documents.")
        texts = (p.get_feature_text() for p in paper_query)
        pbar = tqdm(zip(texts, paper_query), total=nu_papers)

        norm_kwds_to_now = {}
        for i, (txt, p) in enumerate(pbar):
            doc = self.nlp(txt)
            lemma_text, kwds = extract_keyword_from_doc(doc)
            p.lemma_text = lemma_text
            for kwd, score, count in kwds:
                if kwd not in norm_kwds_to_now:
                    db_kwd = Keyword(kwd)
                    norm_kwds_to_now[kwd] = db_kwd
                else:  # Use existing keyword in database if it exists
                    db_kwd = norm_kwds_to_now[kwd]
                with session.no_autoflush:
                    # Make sure not to flush when PaperKeywords has no primary keys
                    assoc = PaperKeywords(raw_keyword=kwd, score=score, count=count)
                    assoc.keyword = db_kwd
                    p.keywords.append(assoc)
            if i % batch_size == 0:
                pbar.set_description(f'Committing up to {i}...')
                session.commit()
                pbar.set_description(f'Committed up to {i}.')

    def get_new_paper_records(self, paper_dict, paper_kwds, pbar):
        """
        Create records for a given paper's new keywords

        Args:
            paper_dict: A dict representation of a paper
            paper_kwds: dict representations of keywords to potentially add to the paper
            pbar: a tqdm progress bar to update

        Returns:
            records: dicts for PaperKeywords to be added for the given Paper dict form
        """
        records = []
        for k in paper_kwds:
            if (
                k["keyword_id"] in paper_dict["keyword_ids"]
            ):  # Don't add keyword if its already there.
                continue
            elif f' {k["raw_keyword"].strip().lower()} ' in paper_dict["lemma_text"]:
                # ras (Royal Astronomical Socieity) will show up when its "contRASt"
                # mil (Acronym for a meteor) with show up with "siMILar"
                # But want "black hole" to show up when "massive black hole" is there.
                # Don't count as containing when it is within a word (needs spaces on sides).
                r = self.get_pk_dict(paper_dict, k)
                records.append(r)
            else:
                continue
        pbar.update(1)
        return records

    def get_missed_keyword_locations(
        self, session, no_below=0, no_above=1.0, npartitions=1000
    ):
        """
        For each keyword, find papers where singlerank did not identify that keyword,
        but the keyword is present in the abstract or title.

        Args:
            session: sqlalchemy session
            no_below: Do not find papers for keywords which appear fewer t
                han this many times
            no_above: Do not find papers for keywords which occur in more than this
                fraction of the corpus
            npartitions: number of partitions for dask delayed computation

        Returns:
            list of dictionaries, each a record to be added to the PaperKeywords table.

        """
        corpus_query = session.query(Paper)
        corpus_size = corpus_query.count()
        no_above_abs = int(no_above * corpus_size)
        pkq = (
            session.query(PaperKeywords)
            .group_by(PaperKeywords.raw_keyword)  # TODO: use keyword_id instead?
            .having(func.count() >= no_below)
            .having(func.count() <= no_above_abs)
        )
        papers = [p.to_dict() for p in corpus_query.all()]
        pks = [pk.to_dict() for pk in pkq.all()]

        def make_batch(papers, paper_kwds, pbar):
            sub_results = []
            for p in papers:
                p_results = self.get_new_paper_records(p, paper_kwds, pbar)
                sub_results.append(p_results)
            return sub_results

        batches = []
        pbar = tqdm(papers)
        batch_size = ceil(len(papers) / npartitions)
        for i in range(0, len(papers), batch_size):
            sub_papers = papers[i: i + batch_size]
            result_batch = dask.delayed(make_batch)(sub_papers, pks, pbar)
            batches.append(result_batch)

        records_batches = dask.compute(*batches)
        records = [
            r for batch in records_batches for paper_set in batch for r in paper_set
        ]
        return records

    @staticmethod
    def get_pk_dict(paper_dict: Dict, paper_kwd: Dict) -> Dict[str, Union[str, int]]:
        """
        Get dictionary record of a new PaperKeyword to be added to the database

        Args:
            paper_dict: Dict representation of Paper
            paper_kwd: Dict representation of a PaperKeyword

        Returns:
            record: dict for PaperKeyword to be added
        """
        count = paper_dict["lemma_text"].count(paper_kwd["raw_keyword"].lower())
        record = {
            "raw_keyword": paper_kwd["raw_keyword"],
            "keyword_id": paper_kwd["keyword_id"],
            "paper_id": paper_dict["id"],
            "count": count,
        }
        return record


class PaperOrganizer:

    def __init__(
        self,
        no_below=5,
        no_above=0.5,
        min_mean_score=0,
        year_min=None,
        year_max=None,
        journal_blacklist=None,
        keyword_blacklist=None,
        use_keyword_count=True,
    ):
        """
        Object to get prepared gensim corpus and dictionary from provided database
        after applying a variety of filters.

        Args:
            no_below: Keep keywords which are contained in at least no_below documents
            no_above: Keep keywords which are contained in no more than no_above documents
                (fraction of total corpus size, not an absolute number).
            min_mean_score: Keep keywords which have mean scores greater than or equal
                to min_mean_score
            journal_blacklist: Remove papers which are in journal_blacklist
            keyword_blacklist: Remove keywords which are in keyword_blacklist
            use_keyword_count: If use_keyword_count is True, the corpus includes the
                counts of the keywords in each document. If use_keyword_count is False,
                the corpus has a count of 1 if the keyword occurs in a given document,
                and a 0 otherwise.
        """
        self.no_below = no_below
        self.no_above = no_above
        self.min_mean_score = min_mean_score
        self.year_min = year_min
        self.year_max = year_max
        self.journal_blacklist = journal_blacklist
        self.keyword_blacklist = keyword_blacklist
        self.use_keyword_count = use_keyword_count

    @property
    def journal_blacklist(self):
        """
        Journals which should not be included in gensim corpus.

        Getter: Gets a list of these journals
        Setter: Sets these journals. Sets as an empty list if None is provided.
        """
        return self.__journal_blacklist

    @journal_blacklist.setter
    def journal_blacklist(self, journal_blacklist):
        if journal_blacklist is None:
            self.__journal_blacklist = []
        else:
            self.__journal_blacklist = journal_blacklist

    @property
    def keyword_blacklist(self):
        """
        Keywords which should not be included in gensim dictionary.

        Getter: Get a list of these keywords
        Setter: If provided a list, sets these keywords as provided list. If provided
            None, sets these keywords to an empty list. If provided a pathlike object,
            assumes the file has one keyword per line, loads these keywords as a list,
            and sets the keywords to that list.
        """
        return self.__keyword_blacklist

    @keyword_blacklist.setter
    def keyword_blacklist(self, keyword_blacklist):
        if keyword_blacklist is None:
            self.__keyword_blacklist = []
        elif type(keyword_blacklist) == list:
            self.__keyword_blacklist = keyword_blacklist
        else:
            p = Path(keyword_blacklist)
            LOG.info(f"Loading keyword_blacklist from {p}")
            if not p.exists():
                raise ValueError(
                    f"{p} does not exist."
                    f"Please provide list of keywords or path to"
                    f"a text file with one keyword per line."
                )
            else:
                with open(p, "r") as f0:
                    self.__keyword_blacklist = [l for l in f0.read().splitlines()]
                LOG.info(
                    f"Loaded {len(self.__keyword_blacklist)} blacklisted keywords."
                    f"{self.__keyword_blacklist[0:3]}"
                    f"...{self.__keyword_blacklist[-3:-1]}"
                )

    def add_journal_blacklist_to_query(self, q):
        """
        Add journal blacklisting to the given query

        Args:
            q: an sqlalchemy query

        Returns:
            q: new query with journal blacklisting included
        """
        for j in self.journal_blacklist:
            q = q.filter(~Paper.bibcode.contains(j))
        return q

    def get_filtered_keywords(self, session, *args):
        """
        From database, get query result for keywords which pass filters
        (but don't execute the query)

        Args:
            session: sqlalchemy session connected to database
            *args: what objects to return from query, must contain Keyword or one of its
                attributes (ex. Keyword.id, Keyword.keyword).

        Returns:
            Query for keywords which pass filters (exluding blacklist which must
            be applied independently).
        """
        if len(args) == 0:
            args = [Keyword, func.count(Keyword.id), func.avg(PaperKeywords.score)]
        corpus_size = session.query(Paper.id).count()
        # TODO: decide whether to count with limiting by journal blacklist
        no_above_abs = int(self.no_above * corpus_size)

        kwd_query = (
            session.query(*args)
            .join(PaperKeywords)
            .join(Paper)
        )
        if self.year_min is not None:
            kwd_query = kwd_query.filter(Paper.year >= self.year_min)
        if self.year_max is not None:
            kwd_query = kwd_query.filter(Paper.year <= self.year_max)
        kwd_query = (
            kwd_query
            .group_by(Keyword.id)
            .order_by(func.avg(PaperKeywords.score).desc())
            .having(func.count() >= self.no_below)
            .having(func.count() <= no_above_abs)
            .having(func.avg(PaperKeywords.score) >= self.min_mean_score)
        )
        kwd_query = self.add_journal_blacklist_to_query(kwd_query)
        return kwd_query

    def get_keyword_batch_records(self, session, kwds_batch, pbar=None):
        """
        Get counts of keywords for each paper in batches of keywords.

        Args:
            session: an sqlalchemy session with connection to the database
            kwds_batch: Batch of keywords for which to papers keyword
                counts or occurences
            pbar: a tqdm loading bar to update

        Returns:
            Tuple of all the papers to keyword to count mappings.
        """
        q = (
            session.query(
                PaperKeywords.paper_id, PaperKeywords.keyword_id, PaperKeywords.count,
            )
            .filter(PaperKeywords.keyword_id.in_(kwds_batch))
            .join(Paper)  # for the journal blacklist removal
        )
        if self.year_max is not None:
            q = q.filter(Paper.year <= self.year_max)
        if self.year_min is not None:
            q = q.filter(Paper.year >= self.year_min)
        q = self.add_journal_blacklist_to_query(q)
        if pbar is not None:
            pbar.update(1)
        return q.all()

    def get_paper_keyword_records(
        self, session, all_kwd_ids, batch_size, in_memory=False
    ):
        """
        Get counts of keywords for each paper

        Args:
            session: an sqlalchemy session with connection to the database
            all_kwd_ids: all keywords to get paper counts for
            batch_size: Number of keywords to query at once. Limited by maximum
                number of variables of your database
            in_memory: If True, load all PaperKeywords into memory and then
                apply filtering. If False, apply filtering as the keywords are queried.

        Returns:
            all_records: papers and their keyword occurrence counts as a list of tuples
                each tuple ~ (paper id, keyword id, count of keyword in given paper).
        """
        if in_memory is True:
            LOG.warning("\"in_memory\" option is deprecated. Only use batches now.")
        num_kwds = len(all_kwd_ids)
        all_records = []
        kwd_batches = range(0, num_kwds, batch_size)
        pbar = tqdm(kwd_batches)
        for i in pbar:
            kwds_batch = all_kwd_ids[i: i + batch_size]
            records = self.get_keyword_batch_records(session, kwds_batch)
            all_records = all_records + records
        return all_records

    def get_corpus_and_dictionary(self, session, batch_size=990, in_memory=False):
        """
        Get gensim corpus and keyword dictionary from the database, applying the
        specified filters.

        Args:
            session: an sqlalchemy session with connection to database
            batch_size: Number of keywods to query at once. Limited by maximum number
                of variables for the database.
            in_memory: If True, load all PaperKeywords into memory and then
                apply filtering. If False, apply filtering as the keywords are queried.

        Returns:
            corpus: a bag of words gensim corpus of documents and their keywords
            dct: a gensim dictionary of the keywords
            corp2paper: a mapping from corpus index to paper ids in the database
            dct2kwd: a mapping from dictionary index to keywords ids in the database
        """
        max_batch_size = 999 - len(self.journal_blacklist)
        if batch_size > max_batch_size:
            raise ValueError(
                f"{batch_size} (batch_size - len(self.journal_blacklist)) "
                f"greater than maximum number of SQLite variables"
            )
        LOG.info("Getting filtered keywords")
        kwd_query = self.get_filtered_keywords(session, Keyword.id, Keyword.keyword)
        all_ki = [(i, k) for i, k in kwd_query.all() if k not in self.keyword_blacklist]
        all_kwd_ids = [i for i, k in all_ki]
        all_records = self.get_paper_keyword_records(
            session, all_kwd_ids, batch_size, in_memory=in_memory
        )

        LOG.info("Getting paper_ids, keyword_ids, counts")
        paper_ids, keyword_ids, counts = zip(*all_records)
        if self.use_keyword_count is False:
            counts = [1] * len(counts)

        ind2sql = {
            "corp2paper": {i: b for i, b in enumerate(set(paper_ids))},
            "dct2kwd": {i: k for i, k in enumerate(set(keyword_ids))},
        }
        sql2ind = {
            "paper2corp": {b: i for i, b in ind2sql["corp2paper"].items()},
            "kwd2dct": {k: i for i, k in ind2sql["dct2kwd"].items()},
        }
        corp_inds = [sql2ind["paper2corp"][b] for b in paper_ids]
        dct_inds = [sql2ind["kwd2dct"][k] for k in keyword_ids]

        LOG.info("Getting gensim corpus.")
        coo_corpus = ((b, k, c) for b, k, c in zip(corp_inds, dct_inds, counts))
        a = np.fromiter(coo_corpus, dtype=[("row", int), ("col", int), ("value", int)])
        mat = coo_matrix((a["value"], (a["row"], a["col"])))
        corpus = Sparse2Corpus(mat, documents_columns=False)

        LOG.info("Getting gensim dictionary.")
        id2word = {sql2ind["kwd2dct"][i]: k for i, k in all_ki}
        dct = Dictionary.from_corpus(corpus, id2word=id2word)

        corp2paper = [(c, p) for c, p in ind2sql["corp2paper"].items()]
        dct2kwd = [(d, k) for d, k in ind2sql["dct2kwd"].items()]

        return corpus, dct, corp2paper, dct2kwd


def get_spacy_nlp(model_name="en_core_web_sm"):
    """
    Get a spacy model with a modified tokenizer which keeps words with hyphens together.
        For example, x-ray will not be split into "x" and "ray".

    Args:
        model_name: Name of spacy model to modify tokenizer for

    Returns:
        nlp: modified spacy model which does not split tokens on hyphens
    """
    nlp = spacy.load(model_name)

    # modify tokenizer infix patterns to not split on hyphen
    # Need to get words like x-ray and gamma-ray
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # EDIT: commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp


# taken from: https://stackoverflow.com/questions/31191727/moving-back-and-forth-between-an-on-disk-database-and-a-fast-in-memory-database/31212306
def copy_database(source_connection, dest_dbname=":memory:", uri=False):
    """
        Return a connection to a new copy of an existing database.
        Raises an sqlite3.OperationalError if the destination already exists.
    """
    script = "".join(source_connection.iterdump())
    dest_conn = sqlite3.connect(dest_dbname, uri=uri)
    dest_conn.executescript(script)
    return dest_conn


@click.group()
def cli():
    """Tool for filling database with ADS metadata and keywords.
    """


@cli.command(short_help="Write JSONlines metadata to database.")
@click.option("--infile", type=Path, help="JSONlines with ADS metadata")
@click.option(
    "--db_loc", type=Path, default=":memory:", help="Location of output SQLite database"
)
def write_ads_to_db(infile, db_loc=":memory:"):
    """
    Write JSONlines metadata to database.
    """
    engine = create_engine(f"sqlite:///{db_loc}")
    BASE.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        t = 0
        with open(infile, "r") as f0:
            for cnt, line in tqdm(enumerate(f0)):
                r = json.loads(line)
                affil = r["nasa_afil"] == "YES"
                p = Paper(
                    bibcode=r["bibcode"],
                    title=r["title"],
                    abstract=r["abstract"],
                    year=r["year"],
                    citation_count=r["citation_count"],
                    nasa_affiliation=affil,
                )
                session.add(p)
                t += 1
        session.commit()
        LOG.info(f"Added {t} papers to database.")
    except:
        session.rollback()
        LOG.warning(f"Aborted adding papers to database.")
        raise
    finally:
        session.close()


@cli.command()
@click.option("--db_loc", type=Path, help="Path to SQlite database")
@click.option("--config_loc", type=Path, help="Path to YAML configuration")
@click.option(
    "--batch_size", type=int, default=1000, help="Batch size for spacy pipeline"
)
@click.option(
    "--n_process", type=int, default=-1, help="Number of process for spacy pipeline"
)
def get_keywords_from_texts(db_loc, config_loc, batch_size=1000, n_process=-1):
    """
    Use SingleRank to extract keywords from titles and abstracts in database.
    """
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)
    try:
        spacy_model_name = config["keyword_extraction"]["spacy_model_name"]
    except KeyError:
        spacy_model_name = "en_core_web_sm"

    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp(spacy_model_name)
        pm = PaperKeywordExtractor(nlp)
        pm.extract_all_keywords(session, batch_size=batch_size, n_process=n_process)
        LOG.info("Commiting keywords to database.")
        session.commit()
        LOG.info(f"Added extracted keywords to papers.")
    except:
        LOG.warning(f"Aborted extracting keywords.")
        session.rollback()
        raise
    finally:
        session.close()


@cli.command()
@click.option("--db_loc", type=Path, help="Path to SQlite database")
@click.option("--config_loc", type=Path, help="Path to YAML configuration")
def add_missed_locations(db_loc, config_loc):
    """
    Go back through the texts to find missed keyword locations.
    """
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)
    no_below = config["keyword_extraction"]["no_below"]
    no_above = config["keyword_extraction"]["no_above"]
    try:
        spacy_model_name = config["keyword_extraction"]["spacy_model_name"]
    except KeyError:
        spacy_model_name = "en_core_web_sm"

    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp(spacy_model_name)
        pm = PaperKeywordExtractor(nlp)
        records = pm.get_missed_keyword_locations(
            session, no_below=no_below, no_above=no_above
        )
        session.commit()
        LOG.info(f"Got records for PaperKeywords to be added.")
    except:
        LOG.warning(f"Aborted finding missed locations.")
        session.rollback()
        raise
    finally:
        session.close()

    # Use engine instead of session for sake of speed.
    # See https://docs.sqlalchemy.org/en/13/faq/performance.html#i-m-inserting-400-000-rows-with-the-orm-and-it-s-really-slow
    engine.execute(PaperKeywords.__table__.insert(), records)
    LOG.info(f"Added {len(records)} missed locations.")


@cli.command()
@click.option("--db_loc", type=Path, help="Path to SQlite database")
@click.option("--config_loc", type=Path, help="Path to YAML configuration")
@click.option(
    "--prepared_data_dir",
    type=Path,
    help="Output directory in which to place files needed for topic modeling.",
)
@click.option(
    "--db_in_memory/--no_db_in_memory",
    default=False,
    help="dump the whole database into memory",
)
@click.option(
    "--in_memory/--no_in_memory",
    default=False,
    help="Load all keywords into memory before filtering",
)
def prepare_for_lda(
    db_loc, config_loc, prepared_data_dir, db_in_memory=False, in_memory=False
):
    """
    Create files necessary for running LDA topic modeling.
    """
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)
    LOG.info(f"Using config: \n {pformat(config)}")

    if db_in_memory is True:
        LOG.info("Dumping database into memory.")
        with closing(sqlite3.connect(db_loc)) as disk_db:
            mem_db = copy_database(disk_db, "file::memory:?cache=shared", uri=True)
        creator = lambda: sqlite3.connect("file::memory:?cache=shared", uri=True)
        engine = create_engine("sqlite://", creator=creator)
    else:
        engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        po = PaperOrganizer(**config["paper_organizer"])
        corpus, dct, corp2paper, dct2kwd = po.get_corpus_and_dictionary(
            session, in_memory=in_memory
        )
    except:
        session.rollback()
        raise
    finally:
        session.close()
        if db_in_memory is True:
            mem_db.close()

    out_corp = prepared_data_dir / "corpus.mm"
    out_dct = prepared_data_dir / "dct.mm"
    out_corp2paper = prepared_data_dir / "corp2paper.json"
    out_dct2kwd = prepared_data_dir / "dct2kwd.json"

    LOG.info(f"Writing corpus, dct, ind2sql, sql2ind to {prepared_data_dir}")
    prepared_data_dir.mkdir(exist_ok=True)
    MmCorpus.serialize(str(out_corp), corpus)
    dct.save(str(out_dct))
    with open(out_corp2paper, "w") as f0:
        json.dump(corp2paper, f0)
    with open(out_dct2kwd, "w") as f0:
        json.dump(dct2kwd, f0)


if __name__ == "__main__":
    cli()
