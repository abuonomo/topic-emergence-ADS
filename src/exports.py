import logging
from pathlib import Path

import click
import pandas as pd
import yaml
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

import db

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db_loc", type=Path)
@click.option("--config_loc", type=Path)
@click.option("--kwd_export_loc", type=Path)
def keywords(
    db_loc,
    config_loc,
    kwd_export_loc,
):
    with open(config_loc, 'r') as f0:
        config = yaml.safe_load(f0)

    engine = create_engine(f"sqlite:///{db_loc}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        po = db.PaperOrganizer(**config['paper_organizer'])
        q = po.get_filtered_keywords(session, db.Keyword.id, db.Keyword.keyword,
                                     func.count(db.Keyword.id))
        LOG.info("Reading keyword counts from database.")
        df = pd.read_sql(q.statement, con=engine)
        drop_inds = df.index[df['keyword'].apply(lambda x: x in po.keyword_blacklist)]
        df = df.drop(drop_inds)
        import ipdb; ipdb.set_trace()
        df = df.sort_values('count_1', ascending=False)
        LOG.info(f"Writing keywords to {kwd_export_loc}")
        df.to_csv(kwd_export_loc)
    except:
        LOG.warning(f"Aborted getting keywords from database.")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    cli()
