import json
from html import unescape

import click
import dask
import logging
import pandas as pd
import spacy
import yaml
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pathlib import Path
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Float
from sqlalchemy import create_engine, func, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from textacy.ke import textrank
from tqdm import tqdm
from typing import Dict, Union

from extract_keywords import strip_tags

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

BASE = declarative_base()


class PaperKeywords(BASE):
    __tablename__ = "paper_keywords"

    paper_bibcode = Column(
        String(19), ForeignKey("papers.bibcode"), primary_key=True, nullable=False
    )
    keyword_id = Column(
        Integer, ForeignKey("keywords.id"), primary_key=True, nullable=False
    )
    score = Column(Float)
    raw_keyword = Column(
        String
    )  # TODO: redundant in current implementation, same as keyword.keyword
    count = Column(Integer)
    paper = relationship("Paper", back_populates="keywords")
    keyword = relationship("Keyword", back_populates="papers")

    def __repr__(self):
        return f'<PaperKeywords(paper_bibcode="{self.paper_bibcode}", keyword.keyword="{self.keyword.keyword}")>'


class Paper(BASE):
    __tablename__ = "papers"

    bibcode = Column(String(19), primary_key=True)
    title = Column(String)
    abstract = Column(String)
    citation_count = Column(Integer)
    year = Column(Integer)
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

    def get_tokens(self):
        tokens0 = [
            [pk.keyword.keyword] * pk.count
            for pk in p.keywords
            if pk.keyword_id in kwd_ids
        ]
        tokens = [t for ts in tokens0 for t in ts]


class Keyword(BASE):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True)
    keyword = Column(String, nullable=False, unique=True)
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
        self.dct = None

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
            topn=999,  # This could cause issues with a huge abstract
            window_size=10,
            edge_weighting="count",
            position_bias=False,
        )
        # Remove keywords which are 1 character long or are numbers
        kwds = [(k, v) for k, v in kwds if (not is_nu_like(k)) and (len(k) > 1)]
        text = " ".join([t.lemma_.lower() for t in doc])
        kwd_counts = [(k, v, text.count(k.lower())) for k, v in kwds]
        return text, kwd_counts

    def get_missed_keyword_locations(self, session, no_below=0, no_above=1.0):
        """
        For each keyword, find papers where singlerank did not identify that keyword,
        but the keyword is present in the abstract or title.

        Args:
            session: sqlalchemy session
            no_below: Do not find papers for keywords which appear fewer t
                han this many times
            no_above: Do not find papers for keywords which occur in more than this
                fraction of the corpus

        Returns:
            list of dictionaries, each a record to be added to the PaperKeywords table.

        """
        corpus_size = session.query(Paper).count()
        no_above_abs = int(no_above * corpus_size)
        pkq = (
            session.query(PaperKeywords)
            .group_by(PaperKeywords.raw_keyword)  # TODO: use keyword_id instead?
            .having(func.count() >= no_below)
            .having(func.count() <= no_above_abs)
        )
        records = []
        with session.no_autoflush:
            pbar = tqdm(pkq, total=pkq.count())
            for paper_kwd in pbar:
                pbar.set_description(paper_kwd.raw_keyword)
                q = (
                    session.query(Paper)
                    .filter(
                        ~Paper.keywords.any(
                            PaperKeywords.keyword_id == paper_kwd.keyword_id
                        )
                    )
                    .filter(
                        or_(
                            Paper.title.contains(paper_kwd.raw_keyword),
                            Paper.abstract.contains(paper_kwd.raw_keyword),
                        )
                    )
                )
                papers = q.all()
                for p in papers:
                    d = self.get_pk_dict(p, paper_kwd)
                    records.append(d)
        return records

    @staticmethod
    def get_pk_dict(p: Paper, paper_kwd: PaperKeywords) -> Dict[str, Union[str, int]]:
        count = p.lemma_text.count(paper_kwd.raw_keyword.lower())
        record = {
            "raw_keyword": paper_kwd.raw_keyword,
            "keyword_id": paper_kwd.keyword.id,
            "paper_bibcode": p.bibcode,
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
    ):
        self.no_below = no_below
        self.no_above = no_above
        self.min_mean_score = min_mean_score
        self.year_min = year_min
        self.year_max = year_max
        self.journal_blacklist = journal_blacklist
        self.dictionary = None
        self.corpus = None

    @staticmethod
    def get_year_counts(session):
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
        for j in self.journal_blacklist:
            years_query = years_query.filter(~Paper.bibcode.contains(j))
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
        years = [y for y, c in self.get_year_counts(session)]
        year_min = min(years)
        year_max = max(years)
        for y in range(year_min, year_max):
            if y in yd:
                c = yd[y]
            else:
                c = 0
            fy[y] = c
        return fy

    def get_filtered_keywords(self, session):
        kwd_query = self._get_filtered_keywords(session)
        return kwd_query.all()

    def _get_filtered_keywords(self, session):
        corpus_size = session.query(Paper).count()
        no_above_abs = int(self.no_above * corpus_size)
        kwd_query = (
            session.query(
                Keyword, func.count(Keyword.id), func.avg(PaperKeywords.score)
            )
            .join(PaperKeywords)
            .join(Paper)
            .group_by(Keyword.id)
            .order_by(func.avg(PaperKeywords.score).desc())
            .having(func.count() >= self.no_below)
            .having(func.count() <= no_above_abs)
            .having(func.avg(PaperKeywords.score) >= self.min_mean_score)
        )
        for j in self.journal_blacklist:
            kwd_query = kwd_query.filter(~Paper.bibcode.contains(j))
        return kwd_query

    def get_tokens(self, session):
        kwds = self.get_filtered_keywords(session)
        q = session.query(Paper)
        for j in self.journal_blacklist:
            q = q.filter(~Paper.bibcode.contains(j))
        kwd_ids = [k.id for k, _, _ in kwds]
        tokens = []
        for p in tqdm(q, total=q.count()):
            tokens0 = [
                [pk.keyword.keyword] * pk.count
                for pk in p.keywords
                if pk.keyword_id in kwd_ids
            ]
            paper_tokens = [t for ts in tokens0 for t in ts]
            tokens.append(paper_tokens)
        return tokens

    def make_all_topic_models(self, session, topic_range, **kwargs):
        # cluster = LocalCluster(silence_logs=False)
        # client = Client(cluster)
        # LOG.info(f"Dask dashboard: {client.dashboard_link}")
        gensim_logger = logging.getLogger("gensim")
        gensim_logger.setLevel(logging.DEBUG)
        LOG.setLevel(logging.DEBUG)
        tokens = self.get_tokens(session)
        self.dictionary = Dictionary(tokens)
        self.corpus = [self.dictionary.doc2bow(ts) for ts in tokens]

        @dask.delayed
        def make_topic_model(n_topics):
            LOG.warning(f"Making topic model with {n_topics} topics.")
            lda = LdaModel(
                self.corpus, id2word=self.dictionary, num_topics=n_topics, **kwargs
            )
            return lda

        jobs = []
        for t in topic_range:
            j = make_topic_model(t)
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

    def set_dictionary(self, session):
        tokens = self.get_tokens(session)
        self.dct = Dictionary(tokens)

    def get_corpus(self, session):
        return [self.dct.doc2bow(d) for d in self.get_tokens(session)]


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
def add_missed_locations(db_loc):
    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp()
        pm = PaperKeywordExtractor(nlp)
        records = pm.get_missed_keyword_locations(session, no_below=100, no_above=0.25)
        session.commit()
        LOG.info(f"Got records for PaperKeywords to be added.")
    except:
        LOG.warning(f"Aborted extracting keywords.")
        session.rollback()
        raise
    finally:
        session.close()

    # Use engine instead of session for sake of speed.
    # See https://docs.sqlalchemy.org/en/13/faq/performance.html#i-m-inserting-400-000-rows-with-the-orm-and-it-s-really-slow
    engine.execute(PaperKeywords.__table__.insert(), records)
    LOG.info(f"Added {len(records)} missed locations.")


@cli.command()
@click.option("--config_loc", type=Path)
@click.option("--out_models_dir", type=Path)
@click.option("--db_loc", type=Path)
def make_topic_models(db_loc, config_loc, out_models_dir):
    with open(config_loc, "r") as f0:
        config = yaml.safe_load(f0)

    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        LOG.info(f"Running with config: {config}")
        po = PaperOrganizer(**config["paper_organizer"])
        models = po.make_all_topic_models(
            session, config["topic_range"], **config["lda"],
        )
        session.commit()
        LOG.info(f"Got records for PaperKeywords to be added.")
    except:
        LOG.warning(f"Aborted extracting keywords.")
        session.rollback()
        raise
    finally:
        session.close()

    out_models_dir.mkdir(exist_ok=True)
    for m in models:
        mp = out_models_dir / f"topic_model{m.num_topics}"
        LOG.info(f"Writing model to {mp}")
        m.save(str(mp))

    od = out_models_dir / "dictionary"
    LOG.info(f"Writing dictionary to {od}")
    po.dictionary.save(str(od))

    oc = out_models_dir / "corpus.mm"
    LOG.info(f"Writing corpus to {oc}")
    MmCorpus.serialize((str(oc)), po.corpus)


@cli.command()
@click.option("--infile", type=Path)
@click.option("--db_loc", type=Path, default=":memory:")
def write_ads_to_db(infile, db_loc=":memory:"):
    engine = create_engine(f"sqlite:///{db_loc}", echo=True)
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
