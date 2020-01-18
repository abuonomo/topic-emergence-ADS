#########
# Predicting Research Trends with Semantic and Neural Networks with an application in Quantum Physics
# Mario Krenn, Anton Zeilinger
# https://arxiv.org/abs/1906.06843 (in PNAS)
#
# This file is the navigator for creating a semantic network from data,
# and using it for predicting future research trends using artificial
# neural networks.
#
# The code is far from clean, but should give the principle idea.
# If you have any questions, dont hasitate to contact me:
# mario.krenn@univie.ac.at / mariokrenn.wordpress.com
#
#
#
# Mario Krenn, Toronto, Canada, 14.01.2020
#
#########

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analyse_network import collaps_network
from calc_properties import calculate_all_network_properties
from create_full_article_data import create_full_data_arxiv
from create_full_article_data_APS import create_full_data_APS
from create_network import create_network
from prepare_ancient_semnets import create_ancient_networks
from prepare_training_data import prepare_training_data
from train_nn import train_nn

from src.join_and_clean import load_records_to_dataframe

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_filled_rows_df(df: pd.DataFrame) -> pd.DataFrame:
    title = df["title"].apply(lambda x: x[0] if type(x) is list else "")
    title = title.astype(str)
    title_not_empty_ind = title.apply(lambda x: x.strip() is not "")
    abstract = df["abstract"].apply(lambda x: x if type(x) is str else "")
    abstract = abstract.astype(str)
    abs_not_empty_ind = abstract.apply(lambda x: x.strip() is not "")
    not_empty_ind = title_not_empty_ind & abs_not_empty_ind
    yta = pd.DataFrame()
    start_year = df.loc[not_empty_ind, "year"].min()
    end_year = df.loc[not_empty_ind, "year"].max()
    yta["year"] = df.loc[not_empty_ind, "year"]
    yta["title"] = title[not_empty_ind]
    yta["abstract"] = abstract[not_empty_ind]
    return yta, (start_year, end_year)


def records_to_lists(in_dir, limit=None):
    df = load_records_to_dataframe(in_dir)
    if limit is not None:
        df = df.sample(limit) # if choose first, only get from one year
    yta, (start_year, end_year) = get_filled_rows_df(df)
    yta_pbar = tqdm(zip(yta["year"].astype(str), yta['title'], yta["abstract"]), total=yta.shape[0])
    records = [[y, t, a] for y, t, a in yta_pbar]
    return records, (start_year, end_year)


if __name__ == "__main__":
    LOG.info("Start building semantic network for Quantum Physics")
    LOG.info("Reading astrophysics data.")
    in_dir = Path("data/raw/")
    all_papers, (start_year, end_year) = records_to_lists(in_dir)

    LOG.info("Start creating initial semantic network")

    keyword_loc = Path('data/keywords_small.txt')
    network_T_full, nn_full, all_KW_full = create_network(all_papers, keyword_loc)
    LOG.info("Finished creating first instance of semantic network")

    synonym_list = Path('data/keywords_syns.txt')
    network_T, nn, all_KW = collaps_network(network_T_full, nn_full, all_KW_full, synonym_list)

    LOG.info(
        "Finished collapsing network (synonyms, empty KWs, ...). Start creating ancient networks."
    )

    all_KW, evolving_nets, evolving_nums = create_ancient_networks(
        network_T, nn, all_KW, start_year, end_year
    )
    LOG.info("Finished creating ancient networks")

    all_properties = calculate_all_network_properties(evolving_nets, evolving_nums)
    LOG.info("Finished calculating network properties for all ancient networks")

    prediction_distance = 5  # how many years into the future are we predicting
    all_data_0, all_data_1 = prepare_training_data(
        evolving_nets, all_properties, prediction_distance, start_year
    )
    LOG.info("Finished preparing training data for neural network")

    train_nn(
        all_data_0, all_data_1, prediction_distance, start_year
    )  # Here we train the neural network, and calculate the ROC & AUC
    LOG.info("Finished training neural networks")

