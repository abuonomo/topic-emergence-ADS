import logging
import click
import pandas as pd
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@click.group()
def cli():
    pass

@click.cli
@click.option("--in_slope_complex", type=Path)
@click.option("--viz_data_loc", type=Path)
@click.option("--in_slope_complex", type=Path)
@click.option("--kwd_export_loc", type=Path)
def keywords(in_slope_complex, viz_data_loc, kwd_export_loc):
    LOG.info("Reading data")
    sc = pd.read_csv(in_slope_complex).loc[: , ['stem', 'count']]

    with open(viz_data_loc, 'r') as f0:
        vd = json.loads(json.load(f0))
    t_df = pd.DataFrame(vd['token.table'])

    rdf = pd.read_json(in_slope_complex, orient='records', lines=True)
    skdf = rdf.loc[:, ['stem', 'keyword_list']]

    LOG.info("Joining data")
    tsc = t_df.set_index('Term').join(sc.set_index('stem'))
    kwd_df = tsc.join(skdf.set_index('stem')).reset_index()
    kwd_df = kwd_df.rename(columns={'index': 'term'})

    LOG.info(f"Writing keywords to {kwd_export_loc}")
    kwd_df.to_csv(kwd_export_loc)


if __name__ == "__main__":
    cli()
