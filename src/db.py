import argparse
import json
import logging

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy import Table, Text
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
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
        return f"<Paper(bibcode=\"{self.bibcode}\", title=\"{self.title}\")>"

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
        return f"<Keyword(keyword=\"{self.keyword}\")>"


class PaperMiner:

    def __init__(self, nlp):
        self.nlp = nlp

    def extract_keywords(self, session):
        papers = session.query(Paper).all()


def main(infile, db_loc=":memory:"):
    engine = create_engine(f"sqlite:///{db_loc}", echo=True)
    BASE.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        t = 0
        with open(infile, 'r') as f0:
            for cnt, line in tqdm(enumerate(f0)):
                r = json.loads(line)
                affil = r['nasa_afil'] == 'YES'
                p = Paper(
                    bibcode=r['bibcode'],
                    title=r['title'],
                    abstract=r['abstract'],
                    year=r['year'],
                    citation_count=r['citation_count'],
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

    # p.keywords.append(Keyword("astronomy"))
    # p.keywords.append(Keyword("neutron star"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("i", help="input txt file")
    parser.add_argument("--db_loc", help="location of database")
    args = parser.parse_args()
    main(args.i, args.db_loc)
