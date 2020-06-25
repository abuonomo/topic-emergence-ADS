import json
from html import unescape
from pprint import pformat
import sqlite3
from functools import reduce
from contextlib import closing

import click
import dask
import logging
import numpy as np
import pandas as pd
import spacy
import yaml
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.matutils import Sparse2Corpus
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from math import ceil
from pathlib import Path
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
from typing import Dict, Union, Generator

from extract_keywords import strip_tags

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

BASE = declarative_base()


class PaperKeywords(BASE):
    __tablename__ = "paper_keywords"

    paper_id = Column(
        Integer, ForeignKey("papers.bibcode"), primary_key=True, nullable=False
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
        text = unescape(text)
        text = strip_tags(text)
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
    try:
        float(s)
        return True
    except ValueError:
        return False


class PaperKeywordExtractor:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_all_keywords(
        self, session, batch_size=100, n_process=-1,
    ):
        # TODO: go back through to find keywords which were missed by singlerank
        papers = session.query(Paper).all()
        LOG.info(f"Extracting keywords from {len(papers)} documents.")
        texts = (p.get_feature_text() for p in papers)
        pipe = self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
        pbar = tqdm(zip(pipe, papers), total=len(papers))

        norm_kwds_to_now = {}
        for doc, p in pbar:
            lemma_text, kwds = self.extract_keyword_from_doc(doc)
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

    @staticmethod
    def extract_keyword_from_doc(doc):
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
        return text, kwd_counts

    def get_new_paper_records(self, paper_dict, paper_kwds, pbar):
        records = []
        for k in paper_kwds:
            if (
                k["keyword_id"] in paper_dict["keyword_ids"]
            ):  # Don't add keyword if its already there.
                continue
            elif k["raw_keyword"].lower() in paper_dict["lemma_text"]:
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
            sub_papers = papers[i : i + batch_size]
            result_batch = dask.delayed(make_batch)(sub_papers, pks, pbar)
            batches.append(result_batch)

        records_batches = dask.compute(*batches)
        records = [
            r for batch in records_batches for paper_set in batch for r in paper_set
        ]
        return records

    @staticmethod
    def get_pk_dict(paper_dict: Dict, paper_kwd: Dict) -> Dict[str, Union[str, int]]:
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
    ):
        self.no_below = no_below
        self.no_above = no_above
        self.min_mean_score = min_mean_score
        self.year_min = year_min
        self.year_max = year_max
        self.journal_blacklist = journal_blacklist
        self.keyword_blacklist = keyword_blacklist

    @property
    def journal_blacklist(self):
        return self.__journal_blacklist

    @journal_blacklist.setter
    def journal_blacklist(self, journal_blacklist):
        if journal_blacklist is None:
            self.__journal_blacklist = []
        else:
            self.__journal_blacklist = journal_blacklist

    @property
    def keyword_blacklist(self):
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
                raise ValueError(f"{p} does not exist."
                                 f"Please provide list of keywords or path to"
                                 f"a text file with one keyword per line.")
            else:
                with open(p, "r") as f0:
                    self.__keyword_blacklist = [l for l in f0.read().splitlines()]
                LOG.info(f"Loaded {len(self.__keyword_blacklist)} blacklisted keywords.")

    def add_journal_blacklist_to_query(self, q):
        for j in self.journal_blacklist:
            q = q.filter(~Paper.bibcode.contains(j))
        return q

    def get_year_counts(self, session):
        q = session.query(Paper.year, func.count(Paper.year))
        return q.group_by(Paper.year).all()

    def get_all_kwd_years(self, session):
        kwd_query = self._get_filtered_keywords(session)
        kwd_ids = (k.id for k, v, c in kwd_query)
        years_query = (
            session.query(Paper.year, PaperKeywords.keyword_id, func.count(Paper.year))
            .join(PaperKeywords)
            .join(Keyword)
            .filter(Keyword.id.in_(kwd_ids))
            .group_by(Keyword.id)
            .group_by(Paper.year)
        )
        years_query = self.add_journal_blacklist_to_query(years_query)
        if self.year_min is not None:
            years_query = years_query.filter(Paper.year >= self.year_min)
        if self.year_max is not None:
            years_query = years_query.filter(Paper.year <= self.year_max)
        years = years_query.all()
        records = [{"year": y[0], "keyword_id": y[1], "count": y[2]} for y in years]
        df = pd.DataFrame(records)
        ydf = df.pivot(index="keyword_id", columns="year", values="count").fillna(0)
        ydf = ydf.loc[:, sorted(ydf.columns)]
        return ydf

    def get_kwd_years(self, session, kwd):
        years_query = (
            session.query(Paper.year, func.count(Paper.year))
            .join(PaperKeywords)
            .join(Keyword)
            .filter(Keyword.keyword == kwd)
            .group_by(Paper.year)
        )
        for j in self.journal_blacklist:
            years_query = years_query.filter(~Paper.bibcode.contains(j))
        years = years_query.all()
        yd = dict(years)
        fy = {}
        for y in range(self.year_min, self.year_max):
            if y in yd:
                c = yd[y]
            else:
                c = 0
            fy[y] = c
        return fy

    def get_filtered_keywords(self, session, *args):
        kwd_query = self._get_filtered_keywords(session, *args)
        return kwd_query.all()

    def _get_filtered_keywords(self, session, *args):
        if len(args) == 0:
            args = [Keyword, func.count(Keyword.id), func.avg(PaperKeywords.score)]
        corpus_size = session.query(Paper.id).count()
        # TODO: decide whether to count with limiting by journal blacklist
        no_above_abs = int(self.no_above * corpus_size)

        kwd_query = (
            session.query(*args)
            .join(PaperKeywords)
            .join(Paper)
            .filter(~Keyword.keyword.in_(self.keyword_blacklist))
            .group_by(Keyword.id)
            .order_by(func.avg(PaperKeywords.score).desc())
            .having(func.count() >= self.no_below)
            .having(func.count() <= no_above_abs)
            .having(func.avg(PaperKeywords.score) >= self.min_mean_score)
        )
        kwd_query = self.add_journal_blacklist_to_query(kwd_query)
        return kwd_query

    def get_tokens(self, session) -> Generator:
        kwds = self.get_filtered_keywords(session)
        q = session.query(Paper)
        for j in self.journal_blacklist:
            q = q.filter(~Paper.bibcode.contains(j))
        kwd_ids = [k.id for k, _, _ in kwds]
        tokens = (self.get_doc_tokens(p, kwd_ids) for p in tqdm(q, total=q.count()))
        return tokens

    @staticmethod
    def get_doc_tokens(p, kwd_ids):
        tokens0 = [
            [pk.keyword.keyword] * pk.count
            for pk in p.keywords
            if pk.keyword_id in kwd_ids
        ]
        paper_tokens = [t for ts in tokens0 for t in ts]
        return paper_tokens

    def get_keyword_batch_records(self, session, kwds_batch, pbar=None):
        q = (
            session.query(
                PaperKeywords.paper_id,
                PaperKeywords.keyword_id,
                PaperKeywords.count,
            )
            .filter(PaperKeywords.keyword_id.in_(kwds_batch))
        )
        # q = self.add_journal_blacklist_to_query(q)
        if pbar is not None:
            pbar.update(1)
        return q.all()

    def get_paper_keyword_records(
        self, session, all_kwd_ids, batch_size, in_memory=False
    ):
        if in_memory is True:
            LOG.info("Loading all paper_keywords into memory, then filtering.")
            all_paper_keywords = session.query(
                PaperKeywords.paper_id,
                PaperKeywords.keyword_id,
                PaperKeywords.count,
            ).all()
            all_records = [
                (b, k, c)
                for b, k, c in all_paper_keywords
                if (k in all_kwd_ids)
                # and not any([j in b for j in self.journal_blacklist])
            ]
        else:
            num_kwds = len(all_kwd_ids)
            all_records = []
            kwd_batches = range(0, num_kwds, batch_size)
            pbar = tqdm(kwd_batches)
            for i in pbar:
                kwds_batch = all_kwd_ids[i : i + batch_size]
                records = self.get_keyword_batch_records(session, kwds_batch)
                all_records = all_records + records
        return all_records

    def get_corpus_and_dictionary(self, session, batch_size=999, in_memory=False):
        if batch_size > 999:
            raise ValueError(
                f"{batch_size} greater than maximum number of SQLite variables"
            )
        LOG.info("Getting filtered keywords")
        kwd_query = self._get_filtered_keywords(session, Keyword.id, Keyword.keyword)
        all_ki = kwd_query.all()
        all_kwd_ids = [i for i, k in all_ki]
        all_records = self.get_paper_keyword_records(
            session, all_kwd_ids, batch_size, in_memory=in_memory
        )

        LOG.info("Getting bibcode, keyword, counts")
        bibs, keyword_ids, counts = zip(*all_records)

        ind2sql = {
            "corp2bib": {i: b for i, b in enumerate(set(bibs))},
            "dct2kwd": {i: k for i, k in enumerate(set(keyword_ids))},
        }
        sql2ind = {
            "bib2corp": {b: i for i, b in ind2sql["corp2bib"].items()},
            "kwd2dct": {k: i for i, k in ind2sql["dct2kwd"].items()},
        }
        corp_inds = [sql2ind["bib2corp"][b] for b in bibs]
        dct_inds = [sql2ind["kwd2dct"][k] for k in keyword_ids]

        LOG.info("Getting gensim corpus.")
        coo_corpus = ((b, k, c) for b, k, c in zip(corp_inds, dct_inds, counts))
        a = np.fromiter(coo_corpus, dtype=[("row", int), ("col", int), ("value", int)])
        mat = coo_matrix((a["value"], (a["row"], a["col"])))
        corpus = Sparse2Corpus(mat, documents_columns=False)

        LOG.info("Getting gensim dictionary.")
        id2word = {sql2ind["kwd2dct"][i]: k for i, k in all_ki}
        dct = Dictionary.from_corpus(corpus, id2word=id2word)

        return corpus, dct, ind2sql, sql2ind


class TopicModeler:
    def __init__(self, dictionary, corpus):
        self.dictionary = dictionary
        self.corpus = corpus

    def make_all_topic_models(self, topic_range, **kwargs):
        gensim_logger = logging.getLogger("gensim")
        gensim_logger.setLevel(logging.DEBUG)
        LOG.setLevel(logging.DEBUG)

        @dask.delayed
        def make_topic_model(n_topics, c, d, **kwargs):
            LOG.warning(f"Making topic model with {n_topics} topics.")
            lda = LdaModel(c, id2word=d, num_topics=n_topics, **kwargs)
            return lda

        jobs = []
        for t in topic_range:
            j = make_topic_model(t, self.corpus, self.dictionary, **kwargs)
            jobs.append(j)
        ldas = dask.compute(jobs)[0]
        return ldas

    def get_coherences(self, lda_models):
        coherences = []
        coh_pbar = tqdm(lda_models)
        for lda_model in coh_pbar:
            coh_pbar.set_description(f"n_topics={lda_model.num_topics}")
            cm = CoherenceModel(
                model=lda_model,
                corpus=self.corpus,
                coherence="u_mass",  # Only u_mass because tokens overlap and are out of order
                dictionary=self.dictionary,
            )
            coherence = cm.get_coherence()  # get coherence value
            coherences.append(coherence)

        return coherences


def get_spacy_nlp():
    nlp = spacy.load("en_core_web_sm")

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
    pass


@cli.command()
@click.option("--db_loc", type=Path)
def get_keywords_from_texts(db_loc):
    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp()
        pm = PaperKeywordExtractor(nlp)
        pm.extract_all_keywords(session)
        session.commit()
        LOG.info(f"Added extracted keywords to papers.")
    except:
        LOG.warning(f"Aborted extracting keywords.")
        session.rollback()
        raise
    finally:
        session.close()


@cli.command()
@click.option("--db_loc", type=Path)
@click.option("--config_loc", type=Path)
def add_missed_locations(db_loc, config_loc):
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)
    no_below = config["extraction"]["no_below"]
    no_above = config["extraction"]["no_above"]

    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp()
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
@click.option("--db_loc", type=Path)
@click.option("--config_loc", type=Path)
@click.option("--prepared_data_dir", type=Path)
@click.option("--db_in_memory/--no_db_in_memory", default=False)
@click.option("--in_memory/--no_in_memory", default=False)
def prepare_for_lda(
    db_loc, config_loc, prepared_data_dir, db_in_memory=False, in_memory=False
):
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
        corpus, dct, ind2sql, sql2ind = po.get_corpus_and_dictionary(
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
    out_ind2sql = prepared_data_dir / "ind2sql.json"
    out_sql2ind = prepared_data_dir / "sql2ind.json"

    LOG.info(f"Writing corpus, dct, ind2sql, sql2ind to {prepared_data_dir}")
    prepared_data_dir.mkdir(exist_ok=True)
    MmCorpus.serialize(str(out_corp), corpus)
    dct.save(str(out_dct))
    with open(out_ind2sql, "w") as f0:
        json.dump(ind2sql, f0)
    with open(out_sql2ind, "w") as f0:
        json.dump(sql2ind, f0)


def read_from_prepared_data(prepared_data_dir):
    in_corp = prepared_data_dir / "corpus.mm"
    in_dct = prepared_data_dir / "dct.mm"
    in_ind2sql = prepared_data_dir / "ind2sql.json"
    in_sql2ind = prepared_data_dir / "sql2ind.json"

    corpus = MmCorpus(str(in_corp))
    dictionary = Dictionary.load(str(in_dct))
    with open(in_ind2sql, "r") as f0:
        ind2sql = json.load(f0)
    with open(in_sql2ind, "r") as f0:
        sql2ind = json.load(f0)

    return corpus, dictionary, ind2sql, sql2ind


@cli.command()
@click.option("--prepared_data_dir", type=Path)
@click.option("--config_loc", type=Path)
@click.option("--out_models_dir", type=Path)
@click.option("--out_coh_csv", type=Path)
def make_topic_models(prepared_data_dir, config_loc, out_models_dir, out_coh_csv):
    corpus, dictionary, ind2sql, sql2ind = read_from_prepared_data(prepared_data_dir)
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)

    LOG.info(f"Running with config: \n {pformat(config)}")
    tm = TopicModeler(dictionary, corpus)
    models = tm.make_all_topic_models(config["topic_range"], **config["lda"])
    coherences = tm.get_coherences(models)
    n_topics = [m.num_topics for m in models]
    df = pd.DataFrame(zip(n_topics, coherences))

    LOG.info(f"Writing coherence scores to {out_coh_csv}.")
    df.columns = ['num_topics', 'coherence (u_mass)']
    df.to_csv(out_coh_csv)

    out_models_dir.mkdir(exist_ok=True)
    for m in models:
        mp = out_models_dir / f"topic_model{m.num_topics}"
        LOG.info(f"Writing model to {mp}")
        m.save(str(mp))


@cli.command()
@click.option("--infile", type=Path)
@click.option("--db_loc", type=Path, default=":memory:")
def write_ads_to_db(infile, db_loc=":memory:"):
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


if __name__ == "__main__":
    cli()
