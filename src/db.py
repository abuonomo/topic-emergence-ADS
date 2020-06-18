import json
import logging
from pathlib import Path
from math import ceil

import click
from html import unescape
from extract_keywords import strip_tags
import spacy
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
from gensim.corpora import Dictionary


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
    raw_keyword = Column(String)
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
    keywords = relationship("PaperKeywords", back_populates="paper")

    def __repr__(self):
        return f'<Paper(bibcode="{self.bibcode}", title="{self.title}")>'

    def get_feature_text(self):
        text = f"{self.title}. {self.abstract}"
        text = unescape(text)
        text = strip_tags(text)
        return text


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


class PaperMiner:
    def __init__(self, nlp):
        self.nlp = nlp
        self.dct = None

    def extract_all_keywords(
        self, session, normalize=None, reorder=False, batch_size=100, n_process=-1,
    ):
        # TODO: go back through to find keywords which were missed by singlerank
        papers = session.query(Paper).all()
        LOG.info(f"Extracting keywords from {len(papers)} documents.")
        texts = (p.get_feature_text() for p in papers)
        pipe = self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
        pbar = tqdm(zip(pipe, papers), total=len(papers))

        norm_kwds_to_now = {}
        for doc, p in pbar:
            kwds = self.extract_keyword_from_doc(doc, normalize, reorder)
            for kwd, score in kwds:
                if kwd not in norm_kwds_to_now:
                    db_kwd = Keyword(kwd)
                    norm_kwds_to_now[kwd] = db_kwd
                else:  # Use existing keyword in database if it exists
                    db_kwd = norm_kwds_to_now[kwd]
                with session.no_autoflush:
                    # Make sure not to flush when PaperKeywords has no primary keys
                    assoc = PaperKeywords(raw_keyword=kwd, score=score)
                    assoc.keyword = db_kwd
                    p.keywords.append(assoc)

    @staticmethod
    def extract_keyword_from_doc(doc, normalize=None, reorder=False):
        # SingleRank parameters
        kwds = textrank(
            doc,
            normalize=normalize,
            topn=999,  # This could cause issues with a huge abstract
            window_size=10,
            edge_weighting="count",
            position_bias=False,
        )
        if reorder:
            if normalize is None:
                text = doc.text
            elif normalize == "lemma":
                text = " ".join([t.lemma_ for t in doc])
            t = []
            for i, (k, v) in enumerate(kwds):
                k_inds = [(i, j) for j in range(len(text)) if text.startswith(k, j)]
                t = t + k_inds
            st = sorted(t, key=lambda x: x[1])
            kwds_sorted = [kwds[i[0]] for i in st]
            return kwds_sorted
        else:
            return kwds

    @staticmethod
    def get_year_counts(session):
        q = session.query(Paper.year, func.count(Paper.year))
        return q.group_by(Paper.year).all()

    @staticmethod
    def get_kwd_years(session, kwd):
        years = (
            session.query(Paper.year, func.count(Paper.year))
            .join(PaperKeywords)
            .join(Keyword)
            .filter(Keyword.keyword == kwd)
            .group_by(Paper.year)
            .all()
        )
        return years

    @staticmethod
    def get_tokens(session, no_below=5, no_above=0.5):
        q = session.query(Paper)
        corpus_size = q.count()
        no_above_abs = int(no_above * corpus_size)
        q2 = (
            session.query(Keyword, func.count(Keyword.id))
            .join(PaperKeywords)
            .join(Paper)
            .group_by(Keyword.id)
            .order_by(func.count(Keyword.id).desc())
            .having(func.count() >= no_below)
            .having(func.count() <= no_above_abs)
        )
        limited_kwds = q2.all()
        tokens = (
            [pk.keyword.keyword for pk in paper.keywords if pk.keyword in limited_kwds]
            for paper in q.all()
        )
        return tokens

    def set_dictionary(self, session, no_below=5, no_above=0.5):
        tokens = self.get_tokens(session, no_below, no_above)
        self.dct = Dictionary(tokens)

    def set_corpus(self, session):
        return [self.dct.doc2bow(d) for d in self.get_tokens(session)]


def get_spacy_nlp():
    nlp = spacy.load("en_core_web_sm")

    # modify tokenizer infix patterns to not split on hyphen
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
@click.option("--normalize", type=str, default=None)
def get_keywords_from_texts(db_loc, normalize):
    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp()
        pm = PaperMiner(nlp)
        pm.extract_all_keywords(session, normalize)
        session.commit()
        LOG.info(f"Added extracted keywords to papers")
    except:
        LOG.warning(f"Aborted extracting keywords.")
        session.rollback()
        raise
    finally:
        session.close()


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
