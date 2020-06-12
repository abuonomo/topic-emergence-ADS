import json
import logging
from pathlib import Path

import click
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from textacy.ke import textrank
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

BASE = declarative_base()

PAPER_KEYWORDS = Table(
    "paper_keywords",
    BASE.metadata,
    Column("paper_bibcode", ForeignKey("papers.bibcode"), primary_key=True),
    Column("keyword_id", ForeignKey("keywords.id"), primary_key=True),
)


class Paper(BASE):
    __tablename__ = "papers"

    bibcode = Column(String(19), primary_key=True)
    title = Column(String)
    abstract = Column(String)
    citation_count = Column(Integer)
    year = Column(Integer)
    nasa_affiliation = Column(Boolean)
    keywords = relationship(
        "Keyword", secondary=PAPER_KEYWORDS, back_populates="papers"
    )

    def __repr__(self):
        return f'<Paper(bibcode="{self.bibcode}", title="{self.title}")>'

    def get_feature_text(self):
        return f"{self.title}. {self.abstract}"


class Keyword(BASE):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True)
    keyword = Column(String, nullable=False, unique=True)
    papers = relationship("Paper", secondary=PAPER_KEYWORDS, back_populates="keywords")

    def __init__(self, keyword):
        self.keyword = keyword

    def __repr__(self):
        return f'<Keyword(keyword=\"{self.keyword}\")>'


class PaperMiner:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_all_keywords(self, session, batch_size=100, n_process=-1):
        papers = session.query(Paper).all()
        LOG.info(f"Extracting keywords from {len(papers)} documents.")
        texts = (p.get_feature_text() for p in papers)
        pbar = tqdm(
            zip(
                self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process), papers
            ),
            total=len(papers),
        )
        for doc, p in pbar:
            kwds_sorted = self.extract_keyword_from_doc(doc)
            for kwd, score in kwds_sorted:
                if session.query(Keyword).filter(Keyword.keyword == kwd).count() == 0:
                    db_kwd = Keyword(kwd)
                else:  # Use existing keyword in database if it exists
                    db_kwd = session.query(Keyword).filter(Keyword.keyword == kwd).first()
                p.keywords.append(db_kwd)

    @staticmethod
    def extract_keyword_from_doc(doc):
        # SingleRank parameters
        kwds = textrank(
            doc,
            normalize=None,
            topn=999,  # This could technically cause issues with a huge abstract
            window_size=10,
            edge_weighting="count",
            position_bias=False,
        )
        text = doc.text
        t = []
        for i, (k, v) in enumerate(kwds):
            k_inds = [(i, j) for j in range(len(text)) if text.startswith(k, j)]
            t = t + k_inds
        st = sorted(t, key=lambda x: x[1])
        kwds_sorted = [kwds[i[0]] for i in st]
        return kwds_sorted


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
def get_keywords_from_texts(db_loc):
    engine = create_engine(f"sqlite:///{db_loc}")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        nlp = get_spacy_nlp()
        pm = PaperMiner(nlp)
        pm.extract_all_keywords(session)
        import ipdb; ipdb.set_trace()
        session.commit()
        LOG.warning(f"Added extracted keywords to papers")
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
